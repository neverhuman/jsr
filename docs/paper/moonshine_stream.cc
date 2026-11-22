#include <moonshine-c-api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

static void transcribe(int32_t model, const fs::path &wav, const fs::path &output_dir) {
    const auto audio = read_staged_wav(wav);
    const auto stream = checked(moonshine_create_stream(model, 0), "create stream");
    checked(moonshine_start_stream(model, stream), "start stream");
    transcript_t *transcript = nullptr;
    for (std::size_t offset = 0; offset < audio.size(); offset += packet_samples) {
        const auto count = std::min(packet_samples, audio.size() - offset);
        checked(moonshine_transcribe_add_audio_to_stream(model, stream, audio.data() + offset,
                                                         count, sample_rate, 0), "push audio");
        if ((offset + count) % update_samples == 0 && offset + count < audio.size()) {
            checked(moonshine_transcribe_stream(model, stream, 0, &transcript), "update transcript");
        }
    }
    checked(moonshine_stop_stream(model, stream), "stop stream");
    checked(moonshine_transcribe_stream(model, stream, 0, &transcript), "flush transcript");
    if (transcript == nullptr) {
        throw std::runtime_error("missing final transcript: " + wav.string());
    }
    const auto path = output_dir / (wav.stem().string() + ".txt");
    std::ofstream output(path);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    for (uint64_t i = 0; i < transcript->line_count; ++i) {
        if (i != 0) {
            output << ' ';
        }
        output << transcript->lines[i].text;
    }
    output << '\n';
    output.close();
    checked(moonshine_free_stream(model, stream), "free stream");
}

int main(int argc, char **argv) {
    const std::string_view size = argc > 1 ? argv[1] : "";
    if (argc != 5 || (size != "tiny" && size != "small")) {
        std::cerr << "usage: moonshine-stream tiny|small MODEL_DIR WAV_DIR OUTPUT_DIR\n";
        return 2;
    }
    try {
        const auto arch = size == "tiny" ? MOONSHINE_MODEL_ARCH_TINY_STREAMING
                                         : MOONSHINE_MODEL_ARCH_SMALL_STREAMING;
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
        for (const auto &wave : waves) {
            transcribe(model, wave, output_dir);
        }
        moonshine_free_transcriber(model);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "moonshine-stream: " << error.what() << '\n';
        return 1;
    }
}
