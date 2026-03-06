import argparse
import sys
from pathlib import Path
from typing import Iterable


SUPPORTED_TARGETS = ["rv1103", "rv1103b", "rv1106", "rv1106b", "rv1126b", "rk2118", "rk3562", "rk3566", "rk3568", "rk3576","rk3588"]
SUPPORTED_QUANT_DTYPES = ["w8a8", "w4a16", "w8a16", "w4a8", "w16a16i", "w16a16i_dfp"]
SUPPORTED_QUANT_ALGOS = ["normal", "mmse", "kl_divergence", "gdq"]
SUPPORTED_QUANT_METHODS = ["layer","channel","group32","group64","group96","group128","group160","group192","group224","group256"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".npy"}


def parse_channel_values(raw: str | None, name: str) -> list[list[float]] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} cannot be empty")
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"{name} must be a comma-separated numeric list, got: {raw}") from e
    return [values]


def collect_dataset_images(dataset_dir: Path, recursive: bool) -> list[Path]:
    it: Iterable[Path]
    if recursive:
        it = dataset_dir.rglob("*")
    else:
        it = dataset_dir.glob("*")
    images = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def write_dataset_txt(dataset_dir: Path, out_txt: Path, recursive: bool) -> Path:
    images = collect_dataset_images(dataset_dir, recursive=recursive)
    if not images:
        raise FileNotFoundError(f"No images/npy files found in dataset dir: {dataset_dir}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        for p in images:
            f.write(str(p.resolve()) + "\n")

    print(f"[INFO] Wrote dataset list: {out_txt}")
    print(f"[INFO] Dataset entries: {len(images)}")
    return out_txt


def validate_dataset_txt(dataset_txt: Path) -> int:
    if not dataset_txt.exists():
        raise FileNotFoundError(f"Dataset txt not found: {dataset_txt}")

    base_dir = dataset_txt.parent
    valid_lines = 0
    with dataset_txt.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths = line.split()
            for raw_path in paths:
                p = Path(raw_path)
                candidates = [p] if p.is_absolute() else [base_dir / p, Path.cwd() / p]
                if not any(c.exists() for c in candidates):
                    raise FileNotFoundError(
                        f"Dataset entry missing at line {line_no}: '{raw_path}' (checked relative to {base_dir} and cwd)"
                    )
            valid_lines += 1

    if valid_lines == 0:
        raise ValueError(f"Dataset txt has no valid entries: {dataset_txt}")

    print(f"[INFO] Validated dataset entries: {valid_lines}")
    return valid_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX model to RKNN")
    parser.add_argument("-i", "--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("-o", "--output", default=None, help="Output RKNN model path")
    parser.add_argument(
        "--target-platform",
        default="rk3588",
        choices=SUPPORTED_TARGETS,
        help="RKNN target platform",
    )

    parser.add_argument("--quantize", action="store_true", help="Enable quantization during build")
    parser.add_argument("--dataset", default=None, help="Path to dataset txt for quantization")
    parser.add_argument("--dataset-dir", default=None, help="Directory of calibration images/npy to auto-generate dataset txt")
    parser.add_argument(
        "--dataset-out",
        default="dataset.txt",
        help="Where to write generated dataset txt when --dataset-dir is used",
    )
    parser.add_argument("--dataset-recursive", action="store_true", help="Recursively scan --dataset-dir")

    parser.add_argument(
        "--mean-values",
        default="0,0,0",
        help="Comma-separated mean values (default: 0,0,0)",
    )
    parser.add_argument(
        "--std-values",
        default="255.0,255.0,255.0",
        help="Comma-separated std values (default: 255,255,255)",
    )

    parser.add_argument("--quantized-dtype", default="w8a8", choices=SUPPORTED_QUANT_DTYPES)
    parser.add_argument("--quantized-algorithm", default="normal", choices=SUPPORTED_QUANT_ALGOS)
    parser.add_argument("--quantized-method", default="channel", choices=SUPPORTED_QUANT_METHODS)
    parser.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--auto-hybrid", action="store_true", help="Enable automatic hybrid quantization")
    parser.add_argument("--compress-weight", action="store_true", help="Enable RKNN weight compression")
    parser.add_argument("--single-core-mode", action="store_true", help="Generate single-core model only")
    parser.add_argument("--rknn-batch-size", type=int, default=None, help="Optional rknn_batch_size for build")
    parser.add_argument(
        "--input-size",
        default=None,
        help=(
            "Input size for load_onnx (overrides ONNX graph shapes). "
            "Single int for square (e.g. 1792), or H,W for rectangular (e.g. 1024,1792). "
            "Examples: --input-size 1792  --input-size 1024,1792"
        ),
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose RKNN logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if args.output:
        out_path = Path(args.output)
    else:
        
        stem = onnx_path.stem
        out_name = f"quant_{stem}.rknn" if args.quantize else f"{stem}.rknn"
        out_path = Path(out_name)

    dataset_path = None
    if args.quantize:
        if args.dataset_dir:
            dataset_path = write_dataset_txt(
                dataset_dir=Path(args.dataset_dir),
                out_txt=Path(args.dataset_out),
                recursive=args.dataset_recursive,
            )
        elif args.dataset:
            dataset_path = Path(args.dataset)
        else:
            raise ValueError("Quantization requires --dataset or --dataset-dir")
        validate_dataset_txt(dataset_path)

    mean_values = parse_channel_values(args.mean_values, "--mean-values")
    std_values = parse_channel_values(args.std_values, "--std-values")

    # RKNN Toolkit2 (including 2.3.2) expects legacy ONNX mapping APIs.
    # Newer ONNX versions moved/changed these symbols, so patch compatibility before RKNN import.
    try:
        import types

        import onnx  # type: ignore

        if not hasattr(onnx, "mapping"):
            onnx.mapping = types.SimpleNamespace()  # type: ignore[attr-defined]

        if hasattr(onnx, "_mapping"):
            mapping_obj = onnx.mapping  # type: ignore[attr-defined]
            has_t2n = hasattr(mapping_obj, "TENSOR_TYPE_TO_NP_TYPE")
            has_n2t = hasattr(mapping_obj, "NP_TYPE_TO_TENSOR_TYPE")

            if not has_t2n:
                _m = onnx._mapping  # type: ignore[attr-defined]
                if hasattr(_m, "TENSOR_TYPE_TO_NP_TYPE"):
                    mapping_obj.TENSOR_TYPE_TO_NP_TYPE = _m.TENSOR_TYPE_TO_NP_TYPE
                elif hasattr(_m, "TENSOR_TYPE_MAP"):
                    legacy_map = {}
                    for k, v in _m.TENSOR_TYPE_MAP.items():
                        np_dtype = getattr(v, "np_dtype", None)
                        if np_dtype is None and isinstance(v, tuple) and len(v) > 0:
                            np_dtype = v[0]
                        if np_dtype is not None:
                            legacy_map[k] = np_dtype
                    mapping_obj.TENSOR_TYPE_TO_NP_TYPE = legacy_map

            if not has_n2t and hasattr(mapping_obj, "TENSOR_TYPE_TO_NP_TYPE"):
                import numpy as np

                reverse_map = {}
                for tensor_t, np_t in mapping_obj.TENSOR_TYPE_TO_NP_TYPE.items():
                    try:
                        dt = np.dtype(np_t)
                        reverse_map[dt] = tensor_t
                        reverse_map[dt.type] = tensor_t
                    except Exception:
                        continue
                mapping_obj.NP_TYPE_TO_TENSOR_TYPE = reverse_map
    except ImportError:
        pass

    try:
        from rknn.api import RKNN
    except ImportError as e:
        raise ImportError(
            "Failed to import rknn.api. Activate your RKNN Toolkit2 environment first."
        ) from e

    rknn = RKNN(verbose=args.verbose)
    try:
        print("[INFO] Configuring RKNN...")
        config_kwargs = {
            "target_platform": args.target_platform,
            "quantized_dtype": args.quantized_dtype,
            "quantized_algorithm": args.quantized_algorithm,
            "quantized_method": args.quantized_method,
            "optimization_level": args.optimization_level,
            "compress_weight": args.compress_weight,
            "single_core_mode": args.single_core_mode,
        }
        if mean_values is not None:
            config_kwargs["mean_values"] = mean_values
        if std_values is not None:
            config_kwargs["std_values"] = std_values

        rknn.config(**config_kwargs)

        input_size_list = None
        if args.input_size is not None:
            parts = [int(d) for d in args.input_size.split(",")]
            if len(parts) == 1:
                h = w = parts[0]
            elif len(parts) == 2:
                h, w = parts
            else:
                raise ValueError("--input-size expects a single int (square) or H,W (rectangular)")
            input_size_list = [[1, 3, h, w]]
            print(f"[INFO] Using input_size_list: {input_size_list}")

        print(f"[INFO] Loading ONNX: {onnx_path}")
        load_kwargs = {"model": str(onnx_path)}
        if input_size_list is not None:
            load_kwargs["input_size_list"] = input_size_list
        ret = rknn.load_onnx(**load_kwargs)
        if ret != 0:
            print("[ERROR] rknn.load_onnx failed")
            return ret

        print("[INFO] Building RKNN model...")
        build_kwargs = {"do_quantization": args.quantize}
        if dataset_path is not None:
            build_kwargs["dataset"] = str(dataset_path)
        if args.rknn_batch_size is not None:
            build_kwargs["rknn_batch_size"] = args.rknn_batch_size
        if args.auto_hybrid:
            build_kwargs["auto_hybrid"] = True

        ret = rknn.build(**build_kwargs)
        if ret != 0:
            print("[ERROR] rknn.build failed")
            return ret

        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Exporting RKNN: {out_path}")
        ret = rknn.export_rknn(str(out_path))
        if ret != 0:
            print("[ERROR] rknn.export_rknn failed")
            return ret

        print("[INFO] RKNN export completed")

        print("[OK] Done")
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
