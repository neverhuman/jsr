// Cold single-stream service around transcribe.cpp (GGUF) for the paper's cold protocol.
//
//   transcribe-cold MODE MODEL.gguf WAV_DIR PREDICTIONS.jsonl THREADS [TRACE.jsonl]
//
// MODE is `offline` (whole utterance, full attention) or `stream:LEFT_MS:CHUNK_MS:RIGHT_MS`
// (buffered streaming: the encoder is re-run over [left | chunk | right] for every chunk).
// One model load per process, the staged u%05d.wav files in name order, one final
// hypothesis per file as `{"id", "hyp"}`. Streaming audio is pushed in 20 ms packets, the
// packet size the other cold services use.
//
// TRACE.jsonl (streaming only) holds one line per changed partial and one final line:
//   id, audio_ms (audio pushed so far), wall_ms (measured wall clock since the stream
//   began in this unpaced run), paced_ms (the time the same text would appear if audio
//   arrived in real time: each packet starts at max(previous finish, its arrival) and
//   adds its measured compute), compute_ms (this push), text, final.
//   audio_s, chunk_wall_ms and cum_wall_ms repeat the same values under the names the
//   word-lag harness reads.

#include <transcribe.h>
#include <transcribe/parakeet.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

static void check(transcribe_status status, const char *operation) {
    if (status != TRANSCRIBE_OK) {
        throw std::runtime_error(std::string(operation) + ": " + transcribe_status_string(status));
    }
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

static void json_string(std::ostream &out, std::string_view text) {
    out << '"';
    for (const unsigned char c : text) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                char escaped[8];
                std::snprintf(escaped, sizeof escaped, "\\u%04x", c);
                out << escaped;
            } else {
                out << char(c);
            }
        }
    }
    out << '"';
}

static void log_warnings(transcribe_log_level level, const char *message, void *) {
    if (level == TRANSCRIBE_LOG_LEVEL_WARN || level == TRANSCRIBE_LOG_LEVEL_ERROR) {
        std::cerr << message;
    }
}

struct Trace {
    std::ofstream out;
    void line(const std::string &id, double audio_ms, double wall_ms, double paced_ms,
              double compute_ms, std::string_view text, bool final) {
        out << "{\"id\":";
        json_string(out, id);
        out << std::fixed << std::setprecision(3) << ",\"audio_ms\":" << audio_ms
            << ",\"wall_ms\":" << wall_ms << ",\"paced_ms\":" << paced_ms
            << ",\"compute_ms\":" << compute_ms << ",\"text\":";
        json_string(out, text);
        out << ",\"final\":" << (final ? "true" : "false") << std::setprecision(4)
            << ",\"audio_s\":" << audio_ms / 1000.0 << std::setprecision(3)
            << ",\"chunk_wall_ms\":" << compute_ms << ",\"cum_wall_ms\":" << wall_ms << "}\n";
    }
};

static std::string current_text(transcribe_session *session) {
    const char *text = transcribe_full_text(session);
    return text == nullptr ? std::string() : std::string(text);
}

static std::string run_offline(transcribe_session *session, const std::vector<float> &audio) {
    transcribe_run_params params;
    transcribe_run_params_init(&params);
    params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    check(transcribe_run(session, audio.data(), int(audio.size()), &params), "run");
    return current_text(session);
}

static std::string run_stream(transcribe_session *session, const std::vector<float> &audio,
                              const std::array<int, 3> &context, const std::string &id, Trace *trace) {
    using clock = std::chrono::steady_clock;
    const auto ms = [](clock::duration span) {
        return std::chrono::duration<double, std::milli>(span).count();
    };
    transcribe_run_params run_params;
    transcribe_run_params_init(&run_params);
    run_params.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
    transcribe_parakeet_buffered_stream_ext ext;
    transcribe_parakeet_buffered_stream_ext_init(&ext);
    ext.left_ms = context[0];
    ext.chunk_ms = context[1];
    ext.right_ms = context[2];
    transcribe_stream_params stream_params;
    transcribe_stream_params_init(&stream_params);
    stream_params.family = &ext.ext;
    const auto begin = clock::now();
    check(transcribe_stream_begin(session, &run_params, &stream_params), "stream begin");
    double paced_ms = ms(clock::now() - begin);
    for (std::size_t offset = 0; offset < audio.size(); offset += packet_samples) {
        const auto count = std::min(packet_samples, audio.size() - offset);
        transcribe_stream_update update;
        transcribe_stream_update_init(&update);
        const auto start = clock::now();
        check(transcribe_stream_feed(session, audio.data() + offset, int(count), &update), "stream feed");
        const auto finish = clock::now();
        const double audio_ms = double(offset + count) * 1000.0 / sample_rate;
        const double compute_ms = ms(finish - start);
        paced_ms = std::max(paced_ms, audio_ms) + compute_ms;
        if (trace != nullptr && update.result_changed) {
            trace->line(id, audio_ms, ms(finish - begin), paced_ms, compute_ms, current_text(session), false);
        }
    }
    transcribe_stream_update update;
    transcribe_stream_update_init(&update);
    const auto start = clock::now();
    check(transcribe_stream_finalize(session, &update), "stream finalize");
    const auto finish = clock::now();
    const double audio_ms = double(audio.size()) * 1000.0 / sample_rate;
    const double compute_ms = ms(finish - start);
    paced_ms = std::max(paced_ms, audio_ms) + compute_ms;
    auto text = current_text(session);
    if (trace != nullptr) {
        trace->line(id, audio_ms, ms(finish - begin), paced_ms, compute_ms, text, true);
    }
    return text;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: transcribe-cold offline|stream:LEFT_MS:CHUNK_MS:RIGHT_MS MODEL.gguf WAV_DIR "
                     "PREDICTIONS.jsonl THREADS [TRACE.jsonl]\n";
        return 2;
    }
    try {
        const std::string mode = argv[1];
        const bool streaming = mode.rfind("stream:", 0) == 0;
        std::array<int, 3> context{};
        if (streaming) {
            if (std::sscanf(mode.c_str(), "stream:%d:%d:%d", &context[0], &context[1], &context[2]) != 3) {
                throw std::runtime_error("invalid streaming mode: " + mode);
            }
        } else if (mode != "offline") {
            throw std::runtime_error("unknown mode: " + mode);
        }
        const int threads = std::atoi(argv[5]);
        if (threads < 1) {
            throw std::runtime_error("THREADS must be at least 1");
        }
        if (argc == 7 && !streaming) {
            throw std::runtime_error("a trace needs a streaming mode");
        }
        std::vector<fs::path> waves;
        for (const auto &entry : fs::directory_iterator(argv[3])) {
            const auto name = entry.path().filename().string();
            if (!entry.is_regular_file() || entry.path().extension() != ".wav" || name.size() != 10 ||
                name[0] != 'u' || !std::all_of(name.begin() + 1, name.begin() + 6, [](unsigned char c) {
                    return c >= '0' && c <= '9';
                })) {
                throw std::runtime_error("invalid staged input: " + entry.path().string());
            }
            waves.push_back(entry.path());
        }
        std::sort(waves.begin(), waves.end());
        if (waves.empty()) {
            throw std::runtime_error("staged WAV directory is empty");
        }
        const fs::path predictions = argv[4];
        if (fs::exists(predictions)) {
            throw std::runtime_error("predictions file already exists: " + predictions.string());
        }
        transcribe_log_set(log_warnings, nullptr);
        transcribe_model_load_params load;
        transcribe_model_load_params_init(&load);
        load.backend = TRANSCRIBE_BACKEND_CPU;
        transcribe_session_params session_params;
        transcribe_session_params_init(&session_params);
        session_params.n_threads = threads;
        transcribe_session *session = nullptr;
        check(transcribe_open(argv[2], &load, &session_params, &session), "open model");
        const auto *model = transcribe_get_model(session);
        if (streaming && !transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
                                                            TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM)) {
            throw std::runtime_error("model does not accept buffered streaming");
        }
        std::cerr << "{\"runtime\":\"transcribe.cpp\",\"version\":";
        json_string(std::cerr, transcribe_version());
        std::cerr << ",\"commit\":";
        json_string(std::cerr, transcribe_version_commit());
        std::cerr << ",\"backend\":";
        json_string(std::cerr, transcribe_model_backend(model));
        std::cerr << ",\"variant\":";
        json_string(std::cerr, transcribe_model_variant_string(model));
        std::cerr << ",\"mode\":";
        json_string(std::cerr, mode);
        std::cerr << ",\"threads\":" << threads << ",\"input_packet_ms\":" << (streaming ? 20 : 0) << "}\n";
        std::ofstream output(predictions);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        Trace trace;
        Trace *trace_ptr = nullptr;
        if (argc == 7) {
            trace.out.open(argv[6]);
            trace.out.exceptions(std::ios::failbit | std::ios::badbit);
            trace_ptr = &trace;
        }
        for (const auto &wave : waves) {
            const auto audio = read_staged_wav(wave);
            const auto id = wave.stem().string();
            const auto text = streaming ? run_stream(session, audio, context, id, trace_ptr)
                                        : run_offline(session, audio);
            output << "{\"id\":";
            json_string(output, id);
            output << ",\"hyp\":";
            json_string(output, text);
            output << "}\n";
        }
        output.close();
        if (trace_ptr != nullptr) {
            trace.out.close();
        }
        transcribe_session_free(session);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "transcribe-cold: " << error.what() << '\n';
        return 1;
    }
}
