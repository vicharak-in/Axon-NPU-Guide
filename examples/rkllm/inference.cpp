#include <cstddef>
#include "rkllm.h"
#include <chrono>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string model_path;
    std::string target_platform = "rk3588";
    std::string prompt;
    std::string role = "user";

    int max_new_tokens = 512;
    int max_context_len = 4096;
    int top_k = 1;
    float top_p = 0.95f;
    float temperature = 0.8f;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int base_domain_id = 0;
    int enabled_cpus_num = 4;

    bool stream = false;
    bool print_perf = false;
    bool keep_history = false;
    bool enable_thinking = false;
    bool embed_flash = true;
    bool keep_special_tokens = false;

    std::string lora_model_path;
    std::string lora_name = "default_lora";
    float lora_scale = 1.0f;

    std::string prompt_cache_load_path;
    std::string prompt_cache_save_path;

    std::string system_prompt;
    std::string chat_template_prefix;
    std::string chat_template_postfix;
};

struct CallbackState {
    bool stream = false;
    bool has_error = false;
    std::string text;
    RKLLMPerfStat perf{};
};

class RKLLMApp {
public:
    explicit RKLLMApp(Options options) : options_(std::move(options)) {}

    ~RKLLMApp() {
        if (handle_ != nullptr) {
            rkllm_destroy(handle_);
            handle_ = nullptr;
        }
    }

    void Init() {
        RKLLMParam param = rkllm_createDefaultParam();
        param.model_path = options_.model_path.c_str();
        param.max_new_tokens = options_.max_new_tokens;
        param.max_context_len = options_.max_context_len;
        param.top_k = options_.top_k;
        param.top_p = options_.top_p;
        param.temperature = options_.temperature;
        param.repeat_penalty = options_.repeat_penalty;
        param.frequency_penalty = options_.frequency_penalty;
        param.presence_penalty = options_.presence_penalty;
        param.skip_special_token = !options_.keep_special_tokens;
        param.is_async = false;
        param.extend_param.base_domain_id = options_.base_domain_id;
        param.extend_param.embed_flash = options_.embed_flash ? 1 : 0;
        param.extend_param.n_batch = 1;
        param.extend_param.use_cross_attn = 0;
        param.extend_param.enabled_cpus_num = static_cast<int8_t>(options_.enabled_cpus_num);
        param.extend_param.enabled_cpus_mask = BuildCpuMask(
            options_.target_platform,
            options_.enabled_cpus_num
        );

        const int ret = rkllm_init(&handle_, &param, Callback);
        if (ret != 0) {
            throw std::runtime_error("rkllm_init failed with code " + std::to_string(ret));
        }

        if (!options_.lora_model_path.empty()) {
            RKLLMLoraAdapter adapter{};
            adapter.lora_adapter_path = options_.lora_model_path.c_str();
            adapter.lora_adapter_name = options_.lora_name.c_str();
            adapter.scale = options_.lora_scale;
            const int lora_ret = rkllm_load_lora(handle_, &adapter);
            if (lora_ret != 0) {
                throw std::runtime_error("rkllm_load_lora failed with code " + std::to_string(lora_ret));
            }
            lora_param_.lora_adapter_name = options_.lora_name.c_str();
            lora_loaded_ = true;
        }

        if (!options_.prompt_cache_load_path.empty()) {
            const int cache_ret = rkllm_load_prompt_cache(handle_, options_.prompt_cache_load_path.c_str());
            if (cache_ret != 0) {
                throw std::runtime_error("rkllm_load_prompt_cache failed with code " + std::to_string(cache_ret));
            }
        }

        if (!options_.system_prompt.empty() ||
            !options_.chat_template_prefix.empty() ||
            !options_.chat_template_postfix.empty()) {
            const int tpl_ret = rkllm_set_chat_template(
                handle_,
                options_.system_prompt.c_str(),
                options_.chat_template_prefix.c_str(),
                options_.chat_template_postfix.c_str()
            );
            if (tpl_ret != 0) {
                throw std::runtime_error("rkllm_set_chat_template failed with code " + std::to_string(tpl_ret));
            }
        }
    }

    std::string Generate(const std::string& prompt, const std::string& role, bool enable_thinking) {
        callback_state_.stream = options_.stream;
        callback_state_.has_error = false;
        callback_state_.text.clear();
        last_generation_wall_time_s_ = 0.0;
        std::memset(&callback_state_.perf, 0, sizeof(callback_state_.perf));

        RKLLMInput input{};
        input.role = role.c_str();
        input.enable_thinking = enable_thinking;
        input.input_type = RKLLM_INPUT_PROMPT;
        input.prompt_input = prompt.c_str();

        RKLLMInferParam infer_param{};
        infer_param.mode = RKLLM_INFER_GENERATE;
        infer_param.keep_history = options_.keep_history ? 1 : 0;
        infer_param.lora_params = lora_loaded_ ? &lora_param_ : nullptr;

        RKLLMPromptCacheParam prompt_cache_param{};
        if (!options_.prompt_cache_save_path.empty()) {
            prompt_cache_param.save_prompt_cache = 1;
            prompt_cache_param.prompt_cache_path = options_.prompt_cache_save_path.c_str();
            infer_param.prompt_cache_params = &prompt_cache_param;
        } else {
            infer_param.prompt_cache_params = nullptr;
        }

        const auto started_at = std::chrono::steady_clock::now();
        const int ret = rkllm_run(handle_, &input, &infer_param, &callback_state_);
        const auto ended_at = std::chrono::steady_clock::now();
        last_generation_wall_time_s_ = std::chrono::duration<double>(ended_at - started_at).count();
        if (ret != 0) {
            throw std::runtime_error("rkllm_run failed with code " + std::to_string(ret));
        }
        if (callback_state_.has_error) {
            throw std::runtime_error("runtime callback reported RKLLM_RUN_ERROR");
        }
        return callback_state_.text;
    }

    void ClearKvCache(bool keep_system_prompt) {
        const int ret = rkllm_clear_kv_cache(handle_, keep_system_prompt ? 1 : 0, nullptr, nullptr);
        if (ret != 0) {
            throw std::runtime_error("rkllm_clear_kv_cache failed with code " + std::to_string(ret));
        }
    }

    void Abort() {
        if (handle_ != nullptr) {
            rkllm_abort(handle_);
        }
    }

    void PrintPerf() const {
        const auto& perf = callback_state_.perf;
        const double decode_tok_per_s =
            perf.generate_time_ms > 0.0f ?
                static_cast<double>(perf.generate_tokens) / (static_cast<double>(perf.generate_time_ms) / 1000.0) :
                0.0;

        const double actual_tok_per_s =
            last_generation_wall_time_s_ > 0.0 ?
                static_cast<double>(perf.generate_tokens) / last_generation_wall_time_s_ :
                0.0;

        std::cout << std::fixed << std::setprecision(2)
                  << "[perf] prefill=" << perf.prefill_time_ms << "ms/" << perf.prefill_tokens
                  << " tok | decode=" << perf.generate_time_ms << "ms/" << perf.generate_tokens
                  << " tok (" << decode_tok_per_s << " tok/s)"
                  << " | actual=" << actual_tok_per_s << " tok/s"
                  << " | mem=" << perf.memory_usage_mb << " MB\n";
        std::cout.unsetf(std::ios::floatfield);
        std::cout << std::setprecision(6);
    }

private:
    static int MaxCpusForPlatform(const std::string& platform) {
        if (platform == "rk3588" || platform == "rk3576") {
            return 8;
        }
        if (platform == "rk3562" || platform == "rv1126b") {
            return 4;
        }
        return 8;
    }

    static uint32_t BuildCpuMask(const std::string& platform, int enabled_cpus_num) {
        int start_cpu = 0;

        if (platform == "rk3588" || platform == "rk3576") {
            start_cpu = 4;  // use BIG cores only
        }

        uint32_t mask = 0;
        for (int cpu = start_cpu; cpu < start_cpu + enabled_cpus_num; ++cpu) {
            mask |= (1u << cpu);
        }
        return mask;
    }

    static int Callback(RKLLMResult* result, void* userdata, LLMCallState state) {
        auto* cb = static_cast<CallbackState*>(userdata);
        if (cb == nullptr) {
            return 0;
        }

        if (state == RKLLM_RUN_ERROR) {
            cb->has_error = true;
            std::cerr << "\n[error] RKLLM runtime callback returned RKLLM_RUN_ERROR\n";
            return 0;
        }

        if (state == RKLLM_RUN_FINISH) {
            if (result != nullptr) {
                cb->perf = result->perf;
            }
            if (cb->stream) {
                std::cout << std::endl;
            }
            return 0;
        }

        if (state == RKLLM_RUN_NORMAL || state == RKLLM_RUN_WAITING) {
            if (result != nullptr && result->text != nullptr) {
                cb->text += result->text;
                if (cb->stream) {
                    std::cout << result->text << std::flush;
                }
            }
        }
        return 0;
    }

    Options options_;
    LLMHandle handle_ = nullptr;
    CallbackState callback_state_{};
    RKLLMLoraParam lora_param_{};
    bool lora_loaded_ = false;
    double last_generation_wall_time_s_ = 0.0;
};

RKLLMApp* g_app = nullptr;

void SignalHandler(int signal_num) {
    if (g_app != nullptr) {
        g_app->Abort();
    }
    std::cerr << "\nInterrupted by signal " << signal_num << ". Exiting.\n";
    std::_Exit(130);
}

void PrintUsage(const char* bin) {
    std::cout
        << "Usage:\n"
        << "  " << bin << " --model <model.rkllm> [--prompt \"Hello\"] [options]\n\n"
        << "Key options:\n"
        << "  --model <path>                Required rkllm model path\n"
        << "  --target-platform <name>      rk3588|rk3576|rk3562|rv1126b (default rk3588)\n"
        << "  --prompt <text>               Single-shot prompt; if omitted, interactive mode\n"
        << "  --stream                      Stream token output in callback\n"
        << "  --print-perf                  Print perf stats after each run\n"
        << "  --keep-history                Keep conversation history (multi-turn)\n"
        << "  --enable-thinking             Set enable_thinking=true in RKLLMInput\n"
        << "  --role <user|tool>            Input role (default user)\n"
        << "  --max-new-tokens <int>        Default 512\n"
        << "  --max-context-len <int>       Default 4096\n"
        << "  --enabled-cpus-num <int>      Default 4; rk3588/rk3576: 1-8, rk3562/rv1126b: 1-4\n"
        << "  --lora-model <path>           Optional LoRA model path\n"
        << "  --lora-name <name>            Optional LoRA adapter name (default default_lora)\n"
        << "  --lora-scale <float>          Optional LoRA scale (default 1.0)\n"
        << "  --prompt-cache-load <path>    Optional prompt cache preload\n"
        << "  --prompt-cache-save <path>    Optional prompt cache save path\n"
        << "  --system-prompt <text>        Optional system prompt for custom template\n"
        << "  --chat-template-prefix <text> Optional custom prompt prefix\n"
        << "  --chat-template-postfix <text> Optional custom prompt postfix\n"
        << "  --help                        Show this help\n\n"
        << "Interactive commands:\n"
        << "  exit                          Exit program\n"
        << "  clear                         Clear KV cache\n";
}

bool NextValue(int argc, char** argv, int& i, std::string& out) {
    if (i + 1 >= argc) {
        return false;
    }
    out = argv[++i];
    return true;
}

Options ParseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--model" && NextValue(argc, argv, i, value)) {
            opt.model_path = value;
        } else if (arg == "--target-platform" && NextValue(argc, argv, i, value)) {
            opt.target_platform = value;
        } else if (arg == "--prompt" && NextValue(argc, argv, i, value)) {
            opt.prompt = value;
        } else if (arg == "--role" && NextValue(argc, argv, i, value)) {
            opt.role = value;
        } else if (arg == "--max-new-tokens" && NextValue(argc, argv, i, value)) {
            opt.max_new_tokens = std::stoi(value);
        } else if (arg == "--max-context-len" && NextValue(argc, argv, i, value)) {
            opt.max_context_len = std::stoi(value);
        } else if (arg == "--top-k" && NextValue(argc, argv, i, value)) {
            opt.top_k = std::stoi(value);
        } else if (arg == "--top-p" && NextValue(argc, argv, i, value)) {
            opt.top_p = std::stof(value);
        } else if (arg == "--temperature" && NextValue(argc, argv, i, value)) {
            opt.temperature = std::stof(value);
        } else if (arg == "--repeat-penalty" && NextValue(argc, argv, i, value)) {
            opt.repeat_penalty = std::stof(value);
        } else if (arg == "--frequency-penalty" && NextValue(argc, argv, i, value)) {
            opt.frequency_penalty = std::stof(value);
        } else if (arg == "--presence-penalty" && NextValue(argc, argv, i, value)) {
            opt.presence_penalty = std::stof(value);
        } else if (arg == "--base-domain-id" && NextValue(argc, argv, i, value)) {
            opt.base_domain_id = std::stoi(value);
        } else if (arg == "--enabled-cpus-num" && NextValue(argc, argv, i, value)) {
            opt.enabled_cpus_num = std::stoi(value);
        } else if (arg == "--lora-model" && NextValue(argc, argv, i, value)) {
            opt.lora_model_path = value;
        } else if (arg == "--lora-name" && NextValue(argc, argv, i, value)) {
            opt.lora_name = value;
        } else if (arg == "--lora-scale" && NextValue(argc, argv, i, value)) {
            opt.lora_scale = std::stof(value);
        } else if (arg == "--prompt-cache-load" && NextValue(argc, argv, i, value)) {
            opt.prompt_cache_load_path = value;
        } else if (arg == "--prompt-cache-save" && NextValue(argc, argv, i, value)) {
            opt.prompt_cache_save_path = value;
        } else if (arg == "--system-prompt" && NextValue(argc, argv, i, value)) {
            opt.system_prompt = value;
        } else if (arg == "--chat-template-prefix" && NextValue(argc, argv, i, value)) {
            opt.chat_template_prefix = value;
        } else if (arg == "--chat-template-postfix" && NextValue(argc, argv, i, value)) {
            opt.chat_template_postfix = value;
        } else if (arg == "--stream") {
            opt.stream = true;
        } else if (arg == "--print-perf") {
            opt.print_perf = true;
        } else if (arg == "--keep-history") {
            opt.keep_history = true;
        } else if (arg == "--enable-thinking") {
            opt.enable_thinking = true;
        } else if (arg == "--no-embed-flash") {
            opt.embed_flash = false;
        } else if (arg == "--keep-special-tokens") {
            opt.keep_special_tokens = true;
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }

    if (opt.model_path.empty()) {
        throw std::runtime_error("missing required argument --model");
    }
    if (opt.role != "user" && opt.role != "tool") {
        throw std::runtime_error("--role must be either 'user' or 'tool'");
    }
    return opt;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        RKLLMApp app(options);
        g_app = &app;
        std::signal(SIGINT, SignalHandler);

        app.Init();

        if (!options.prompt.empty()) {
            if (!options.stream) {
                std::cout << "assistant: ";
            }
            const std::string answer = app.Generate(options.prompt, options.role, options.enable_thinking);
            if (!options.stream) {
                std::cout << answer << '\n';
            }
            if (options.print_perf) {
                app.PrintPerf();
            }
            return 0;
        }

        std::cout << "RKLLM interactive mode. Commands: exit, clear\n";
        while (true) {
            std::cout << "\nuser: ";
            std::string line;
            if (!std::getline(std::cin, line)) {
                break;
            }

            if (line.empty()) {
                continue;
            }
            if (line == "exit") {
                break;
            }
            if (line == "clear") {
                app.ClearKvCache(!options.system_prompt.empty());
                std::cout << "KV cache cleared.\n";
                continue;
            }

            if (!options.stream) {
                std::cout << "assistant: ";
            }
            const std::string answer = app.Generate(line, options.role, options.enable_thinking);
            if (!options.stream) {
                std::cout << answer << '\n';
            }
            if (options.print_perf) {
                app.PrintPerf();
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
