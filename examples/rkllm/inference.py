import argparse
import ctypes
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional


LLMHandle = ctypes.c_void_p

RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3

RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0

TARGET_PLATFORMS = ("rk3576", "rk3588", "rk3562", "rv1126b")


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


LLMResultCallbackType = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)


def cpu_mask_for_platform(platform: str, enabled_cpus_num: int) -> int:
    if platform in {"rk3576", "rk3588"}:
        max_cpus = 8
    elif platform in {"rk3562", "rv1126b"}:
        max_cpus = 4
    else:
        max_cpus = 8

    if enabled_cpus_num <= 0 or enabled_cpus_num > max_cpus:
        raise ValueError(
            f"enabled-cpus-num must be in the range [1, {max_cpus}] for platform {platform}"
        )

    mask = 0
    for cpu in range(enabled_cpus_num):
        mask |= 1 << cpu
    return mask


def load_runtime_library(explicit_path: Optional[str]) -> ctypes.CDLL:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path("librkllmrt.so"),
            Path("lib/librkllmrt.so"),
            Path("rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so"),
            Path("rknn-llm/rkllm-runtime/Linux/librkllm_api/lib/librkllmrt.so"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate.resolve()))
    if explicit_path:
        return ctypes.CDLL(explicit_path)
    return ctypes.CDLL("librkllmrt.so")


class RKLLMRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._text_chunks: List[str] = []
        self._last_perf: Optional[RKLLMPerfStat] = None
        self._last_generation_wall_time_s: Optional[float] = None
        self._lora_param_ref = None
        self._prompt_cache_param_ref = None
        self.handle = LLMHandle()
        self.lib = load_runtime_library(args.runtime_lib)
        self._bind_functions()
        self._create_callback()
        self._init_model()
        if args.lora_model:
            self._load_lora(args.lora_model, args.lora_adapter_name)
        if args.prompt_cache_load:
            self._load_prompt_cache(args.prompt_cache_load)
        if args.chat_template_prefix or args.chat_template_postfix or args.system_prompt:
            self._set_chat_template(
                args.system_prompt or "",
                args.chat_template_prefix or "",
                args.chat_template_postfix or "",
            )
        self.infer_param = self._build_infer_param()

    def _bind_functions(self) -> None:
        self.lib.rkllm_createDefaultParam.argtypes = []
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam

        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(LLMHandle),
            ctypes.POINTER(RKLLMParam),
            LLMResultCallbackType,
        ]
        self.lib.rkllm_init.restype = ctypes.c_int

        self.lib.rkllm_run.argtypes = [
            LLMHandle,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self.lib.rkllm_run.restype = ctypes.c_int

        self.lib.rkllm_destroy.argtypes = [LLMHandle]
        self.lib.rkllm_destroy.restype = ctypes.c_int

        self.lib.rkllm_clear_kv_cache.argtypes = [
            LLMHandle,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.rkllm_clear_kv_cache.restype = ctypes.c_int

        self.lib.rkllm_abort.argtypes = [LLMHandle]
        self.lib.rkllm_abort.restype = ctypes.c_int

        self.lib.rkllm_set_chat_template.argtypes = [
            LLMHandle,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.lib.rkllm_set_chat_template.restype = ctypes.c_int

        self.lib.rkllm_load_lora.argtypes = [LLMHandle, ctypes.POINTER(RKLLMLoraAdapter)]
        self.lib.rkllm_load_lora.restype = ctypes.c_int

        self.lib.rkllm_load_prompt_cache.argtypes = [LLMHandle, ctypes.c_char_p]
        self.lib.rkllm_load_prompt_cache.restype = ctypes.c_int

    def _create_callback(self) -> None:
        def _callback(result_ptr, _userdata, state):
            if state == RKLLM_RUN_FINISH:
                result = result_ptr.contents
                self._last_perf = result.perf
                if self.args.stream:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                return 0

            if state == RKLLM_RUN_ERROR:
                print("RKLLM runtime reported an inference error.", file=sys.stderr)
                return 0

            if state not in (RKLLM_RUN_NORMAL, RKLLM_RUN_WAITING):
                return 0

            result = result_ptr.contents
            if not result.text:
                return 0
            chunk = ctypes.string_at(result.text).decode("utf-8", errors="ignore")
            if not chunk:
                return 0
            self._text_chunks.append(chunk)
            if self.args.stream:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            return 0

        self._callback = LLMResultCallbackType(_callback)

    def _init_model(self) -> None:
        param = self.lib.rkllm_createDefaultParam()
        param.model_path = str(Path(self.args.model).resolve()).encode("utf-8")
        param.max_new_tokens = self.args.max_new_tokens
        param.max_context_len = self.args.max_context_len
        param.top_k = self.args.top_k
        param.top_p = self.args.top_p
        param.temperature = self.args.temperature
        param.repeat_penalty = self.args.repeat_penalty
        param.frequency_penalty = self.args.frequency_penalty
        param.presence_penalty = self.args.presence_penalty
        param.skip_special_token = not self.args.keep_special_tokens
        param.is_async = False
        param.extend_param.base_domain_id = self.args.base_domain_id
        param.extend_param.embed_flash = 1 if self.args.embed_flash else 0
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = self.args.enabled_cpus_num
        param.extend_param.enabled_cpus_mask = cpu_mask_for_platform(
            self.args.target_platform, self.args.enabled_cpus_num
        )

        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), self._callback)
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")

    def _load_lora(self, lora_path: str, lora_name: str) -> None:
        adapter = RKLLMLoraAdapter()
        adapter.lora_adapter_path = str(Path(lora_path).resolve()).encode("utf-8")
        adapter.lora_adapter_name = lora_name.encode("utf-8")
        adapter.scale = self.args.lora_scale
        ret = self.lib.rkllm_load_lora(self.handle, ctypes.byref(adapter))
        if ret != 0:
            raise RuntimeError(f"rkllm_load_lora failed with code {ret}")
        lora_param = RKLLMLoraParam()
        lora_param.lora_adapter_name = lora_name.encode("utf-8")
        self._lora_param_ref = lora_param

    def _load_prompt_cache(self, prompt_cache_path: str) -> None:
        ret = self.lib.rkllm_load_prompt_cache(
            self.handle, str(Path(prompt_cache_path).resolve()).encode("utf-8")
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_load_prompt_cache failed with code {ret}")

    def _set_chat_template(self, system_prompt: str, prefix: str, postfix: str) -> None:
        ret = self.lib.rkllm_set_chat_template(
            self.handle,
            system_prompt.encode("utf-8"),
            prefix.encode("utf-8"),
            postfix.encode("utf-8"),
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_set_chat_template failed with code {ret}")

    def _build_infer_param(self) -> RKLLMInferParam:
        infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        infer_param.mode = RKLLM_INFER_GENERATE
        infer_param.keep_history = 1 if self.args.keep_history else 0
        if self._lora_param_ref is not None:
            infer_param.lora_params = ctypes.pointer(self._lora_param_ref)
        else:
            infer_param.lora_params = None

        if self.args.prompt_cache_save:
            cache_param = RKLLMPromptCacheParam()
            cache_param.save_prompt_cache = 1
            cache_param.prompt_cache_path = str(Path(self.args.prompt_cache_save).resolve()).encode(
                "utf-8"
            )
            self._prompt_cache_param_ref = cache_param
            infer_param.prompt_cache_params = ctypes.pointer(self._prompt_cache_param_ref)
        else:
            infer_param.prompt_cache_params = None
        return infer_param

    def generate(self, prompt: str, role: str = "user", enable_thinking: bool = False) -> str:
        self._text_chunks = []
        self._last_perf = None
        self._last_generation_wall_time_s = None

        rk_input = RKLLMInput()
        ctypes.memset(ctypes.byref(rk_input), 0, ctypes.sizeof(RKLLMInput))
        role_bytes = role.encode("utf-8")
        prompt_bytes = prompt.encode("utf-8")
        rk_input.role = role_bytes
        rk_input.enable_thinking = bool(enable_thinking)
        rk_input.input_type = RKLLM_INPUT_PROMPT
        rk_input.input_data.prompt_input = prompt_bytes

        started_at = time.perf_counter()
        ret = self.lib.rkllm_run(
            self.handle,
            ctypes.byref(rk_input),
            ctypes.byref(self.infer_param),
            None,
        )
        self._last_generation_wall_time_s = time.perf_counter() - started_at
        if ret != 0:
            raise RuntimeError(f"rkllm_run failed with code {ret}")
        return "".join(self._text_chunks)

    def clear_kv_cache(self, keep_system_prompt: bool = True) -> None:
        ret = self.lib.rkllm_clear_kv_cache(
            self.handle, 1 if keep_system_prompt else 0, None, None
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_clear_kv_cache failed with code {ret}")

    def abort(self) -> None:
        self.lib.rkllm_abort(self.handle)

    def close(self) -> None:
        if self.handle:
            self.lib.rkllm_destroy(self.handle)
            self.handle = LLMHandle()

    def print_perf(self) -> None:
        if self._last_perf is None:
            return
        perf = self._last_perf
        decode_tok_per_s = 0.0
        if perf.generate_time_ms > 0:
            decode_tok_per_s = perf.generate_tokens / (perf.generate_time_ms / 1000.0)
        actual_tok_per_s = 0.0
        if self._last_generation_wall_time_s and self._last_generation_wall_time_s > 0:
            actual_tok_per_s = perf.generate_tokens / self._last_generation_wall_time_s
        print(
            "\n[perf] prefill={:.2f}ms/{} tok | decode={:.2f}ms/{} tok ({:.2f} tok/s) | actual={:.2f} tok/s | mem={:.2f} MB".format(
                perf.prefill_time_ms,
                perf.prefill_tokens,
                perf.generate_time_ms,
                perf.generate_tokens,
                decode_tok_per_s,
                actual_tok_per_s,
                perf.memory_usage_mb,
            )
        )


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RKLLM runtime inference script (SDK 1.2.3 style init/run/destroy flow)"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the .rkllm model file",
    )
    parser.add_argument(
        "--runtime-lib",
        help="Path to librkllmrt.so (optional if discoverable via default paths or LD_LIBRARY_PATH)",
    )
    parser.add_argument(
        "--target-platform",
        type=str.lower,
        choices=TARGET_PLATFORMS,
        default="rk3588",
        help="Target Rockchip platform",
    )
    parser.add_argument("--max-new-tokens", type=positive_int, default=512)
    parser.add_argument("--max-context-len", type=positive_int, default=4096)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repeat-penalty", type=positive_float, default=1.1)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--base-domain-id", type=int, default=0)
    parser.add_argument(
        "--enabled-cpus-num",
        type=int,
        default=4,
        help="Number of CPUs: rk3588/rk3576: 1-8, rk3562/rv1126b: 1-4",
    )
    parser.add_argument(
        "--embed-flash",
        action="store_true",
        default=True,
        help="Enable embed_flash in RKLLMExtendParam",
    )
    parser.add_argument(
        "--no-embed-flash",
        action="store_false",
        dest="embed_flash",
        help="Disable embed_flash in RKLLMExtendParam",
    )
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help="Do not skip special tokens in output",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="Enable multi-turn history retention (keep_history=1)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated tokens to stdout during inference",
    )
    parser.add_argument(
        "--print-perf",
        action="store_true",
        help="Print RKLLM perf stats at the end of each run",
    )
    parser.add_argument(
        "--prompt",
        help="Single-shot prompt. If not set, run interactive mode.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Set enable_thinking=true in RKLLMInput (Qwen3 thinking mode)",
    )
    parser.add_argument(
        "--role",
        default="user",
        choices=("user", "tool"),
        help="Role field for RKLLMInput",
    )
    parser.add_argument("--lora-model", help="Path to LoRA model file")
    parser.add_argument(
        "--lora-adapter-name",
        default="default_lora",
        help="Name used to register and select LoRA adapter",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--prompt-cache-load", help="Path of prompt cache file to preload")
    parser.add_argument("--prompt-cache-save", help="Path to save prompt cache during inference")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--chat-template-prefix", default="")
    parser.add_argument("--chat-template-postfix", default="")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")
    if args.lora_model and not Path(args.lora_model).exists():
        raise SystemExit(f"LoRA file not found: {args.lora_model}")
    if args.prompt_cache_load and not Path(args.prompt_cache_load).exists():
        raise SystemExit(f"Prompt cache file not found: {args.prompt_cache_load}")
    if args.target_platform in {"rk3576", "rk3588"}:
        max_cpus = 8
    elif args.target_platform in {"rk3562", "rv1126b"}:
        max_cpus = 4
    else:
        max_cpus = 8

    if args.enabled_cpus_num < 1 or args.enabled_cpus_num > max_cpus:
        raise SystemExit(
            f"--enabled-cpus-num must be in the range [1, {max_cpus}] for platform {args.target_platform}"
        )


def run_interactive(runner: RKLLMRunner, args: argparse.Namespace) -> None:
    print("RKLLM interactive mode. Commands: 'exit', 'clear'")
    while True:
        try:
            user_input = input("\nuser: ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "clear":
            runner.clear_kv_cache(keep_system_prompt=bool(args.system_prompt))
            print("KV cache cleared.")
            continue

        if not args.stream:
            print("assistant: ", end="", flush=True)
        answer = runner.generate(
            prompt=user_input,
            role=args.role,
            enable_thinking=args.enable_thinking,
        )
        if not args.stream:
            print(answer)
        if args.print_perf:
            runner.print_perf()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    runner = None

    def _signal_handler(_sig, _frame):
        nonlocal runner
        if runner is not None:
            runner.abort()
            runner.close()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        runner = RKLLMRunner(args)
        if args.prompt:
            if not args.stream:
                print("assistant: ", end="", flush=True)
            answer = runner.generate(
                prompt=args.prompt,
                role=args.role,
                enable_thinking=args.enable_thinking,
            )
            if not args.stream:
                print(answer)
            if args.print_perf:
                runner.print_perf()
        else:
            run_interactive(runner, args)
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    main()
