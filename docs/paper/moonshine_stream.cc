#include <moonshine-c-api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

constexpr int sample_rate = 16000;
constexpr std::size_t packet_samples = sample_rate / 50;
constexpr std::size_t update_samples = sample_rate / 5;

static int32_t checked(int32_t result, const char *operation) {
    if (result < 0) {
        throw std::runtime_error(std::string(operation) + ": " + moonshine_error_to_string(result));
    }
    return result;
}

static std::vector<float> read_staged_wav(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    std::array<unsigned char, 44> header{};
    input.read(reinterpret_cast<char *>(header.data()), header.size());
    const auto bytes = [&](std::size_t at, int count) {
        uint32_t value = 0;
        for (int i = 0; i < count; ++i) {
            value |= uint32_t(header[at + i]) << (8 * i);
        }
        return value;
    };
    const auto tag = [&](std::size_t at, std::string_view expected) {
        return expected == std::string_view(reinterpret_cast<const char *>(header.data() + at), expected.size());
    };
    const auto size = bytes(40, 4);
    if (!input || !tag(0, "RIFF") || !tag(8, "WAVEfmt ") || !tag(36, "data") ||
        bytes(16, 4) != 16 || bytes(20, 2) != 1 || bytes(22, 2) != 1 ||
        bytes(24, 4) != sample_rate || bytes(34, 2) != 16 || size == 0 || size % 2 != 0 ||
        size + uint64_t{44} != fs::file_size(path)) {
        throw std::runtime_error("expected a staged 16 kHz mono PCM16 WAV: " + path.string());
    }
    std::vector<unsigned char> pcm(size);
    if (!input.read(reinterpret_cast<char *>(pcm.data()), size)) {
        throw std::runtime_error("truncated audio: " + path.string());
    }
    std::vector<float> audio;
    audio.reserve(size / 2);
    for (uint32_t i = 0; i < size; i += 2) {
        const int value = int(pcm[i]) | int(pcm[i + 1]) << 8;
        audio.push_back(float(value >= 32768 ? value - 65536 : value) / 32768.0f);
    }
    return audio;
}

static std::string join_transcript(const transcript_t *transcript) {
    if (transcript == nullptr) return "";
    std::string text;
    for (uint64_t i = 0; i < transcript->line_count; ++i) {
        if (i != 0) text.push_back(' ');
        if (transcript->lines[i].text != nullptr) text.append(transcript->lines[i].text);
    }
    return text;
}

static void write_json_escaped(std::ostream &out, const std::string &s) {
    out << '"';
    for (unsigned char c : s) {
        if (c == '"') out << "\\\"";
        else if (c == '\\') out << "\\\\";
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c >= 0x20) out << char(c);
    }
    out << '"';
}

static void transcribe(int32_t model, const fs::path &wav, const fs::path &output_dir, std::ofstream *trace) {
    using clock = std::chrono::steady_clock;
    const auto audio = read_staged_wav(wav);
    const auto stream = checked(moonshine_create_stream(model, 0), "create stream");
    checked(moonshine_start_stream(model, stream), "start stream");
    transcript_t *transcript = nullptr;
    double cum_wall_ms = 0.0;
    const std::string id = wav.stem().string();
    for (std::size_t offset = 0; offset < audio.size(); offset += packet_samples) {
        const auto t0 = clock::now();
        const auto count = std::min(packet_samples, audio.size() - offset);
        checked(moonshine_transcribe_add_audio_to_stream(model, stream, audio.data() + offset,
                                                         count, sample_rate, 0), "push audio");
        bool updated = false;
        if ((offset + count) % update_samples == 0 && offset + count < audio.size()) {
            checked(moonshine_transcribe_stream(model, stream, 0, &transcript), "update transcript");
            updated = true;
        }
        const double dt_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        cum_wall_ms += dt_ms;
        if (trace != nullptr && updated) {
            *trace << "{\"id\":";
            write_json_escaped(*trace, id);
            *trace << ",\"audio_s\":" << std::fixed << std::setprecision(4)
                   << double(offset + count) / double(sample_rate)
                   << ",\"chunk_wall_ms\":" << std::setprecision(3) << dt_ms
                   << ",\"cum_wall_ms\":" << cum_wall_ms
                   << ",\"text\":";
            write_json_escaped(*trace, join_transcript(transcript));
            *trace << ",\"final\":false}\n";
        }
    }
    const auto t_fin = clock::now();
    checked(moonshine_stop_stream(model, stream), "stop stream");
    checked(moonshine_transcribe_stream(model, stream, 0, &transcript), "flush transcript");
    const double fin_ms = std::chrono::duration<double, std::milli>(clock::now() - t_fin).count();
    cum_wall_ms += fin_ms;
    if (transcript == nullptr) {
        throw std::runtime_error("missing final transcript: " + wav.string());
    }
    const std::string final_text = join_transcript(transcript);
    if (trace != nullptr) {
        *trace << "{\"id\":";
        write_json_escaped(*trace, id);
        *trace << ",\"audio_s\":" << std::fixed << std::setprecision(4)
               << double(audio.size()) / double(sample_rate)
               << ",\"chunk_wall_ms\":" << std::setprecision(3) << fin_ms
               << ",\"cum_wall_ms\":" << cum_wall_ms
               << ",\"text\":";
        write_json_escaped(*trace, final_text);
        *trace << ",\"final\":true}\n";
    }
    const auto path = output_dir / (id + ".txt");
    std::ofstream output(path);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << final_text << '\n';
    output.close();
    checked(moonshine_free_stream(model, stream), "free stream");
}

int main(int argc, char **argv) {
    const std::string_view size = argc > 1 ? argv[1] : "";
    if ((argc != 5 && argc != 6) || (size != "tiny" && size != "small" && size != "medium")) {
        std::cerr << "usage: moonshine-stream tiny|small|medium MODEL_DIR WAV_DIR OUTPUT_DIR [TRACE_JSONL]\n";
        return 2;
    }
    try {
        const auto arch = size == "tiny" ? MOONSHINE_MODEL_ARCH_TINY_STREAMING
                        : size == "small" ? MOONSHINE_MODEL_ARCH_SMALL_STREAMING
                                          : MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
        if (setenv("MOONSHINE_ORT_SINGLE_THREAD", "1", 1) != 0) {
            throw std::runtime_error("could not set single-thread ORT mode");
        }
        const moonshine_option_t options[] = {{"ort_providers", "CPU"}};
        const auto model = checked(moonshine_load_transcriber_from_files(
            argv[2], arch, options, 1, MOONSHINE_HEADER_VERSION), "load model");
        std::vector<fs::path> waves;
        for (const auto &entry : fs::directory_iterator(argv[3])) {
            if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
                throw std::runtime_error("unexpected staged input: " + entry.path().string());
            }
            waves.push_back(entry.path());
        }
        if (waves.empty()) {
            throw std::runtime_error("no staged WAV inputs");
        }
        std::sort(waves.begin(), waves.end());
        const fs::path output_dir(argv[4]);
        fs::create_directory(output_dir);
        if (!fs::is_empty(output_dir)) {
            throw std::runtime_error("output directory must be empty: " + output_dir.string());
        }
        std::ofstream trace_file;
        std::ofstream *trace_ptr = nullptr;
        if (argc == 6) {
            trace_file.open(argv[5]);
            trace_file.exceptions(std::ios::failbit | std::ios::badbit);
            trace_ptr = &trace_file;
        }
        for (const auto &wave : waves) {
            transcribe(model, wave, output_dir, trace_ptr);
        }
        moonshine_free_transcriber(model);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "moonshine-stream: " << error.what() << '\n';
        return 1;
    }
}
