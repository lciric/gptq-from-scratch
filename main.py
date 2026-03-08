"""
GPTQ from scratch — main entry point.

Usage:
    python main.py --model gpt2 --baseline              # Evaluate FP16 baseline perplexity
    python main.py --model gpt2 --quantize --bits 4     # Run GPTQ quantization + evaluate
    python main.py --model gpt2 --quantize --bits 3 --naive  # Naive RTN for comparison
    python main.py --model facebook/opt-1.3b --quantize --bits 4
    python main.py --model meta-llama/Meta-Llama-3-8B --quantize --bits 4 --token YOUR_TOKEN
    python main.py --quantize --bits 4 --group-size 128 --act-order --true-sequential
    python main.py --quantize --bits 4 --all-tricks     # Enable all optimizations
    python main.py --quantize --bits 4 --calib-dataset c4  # C4 calibration (paper setup)
    python main.py --wandb --quantize --bits 4          # Log to Weights & Biases
"""

import argparse
import time
import torch

from model_utils import load_model
from arch_config import get_arch_config
from evaluate import evaluate_perplexity


def parse_args():
    parser = argparse.ArgumentParser(description="GPTQ from scratch")
    parser.add_argument(
        "--model", type=str, default="gpt2", help="HuggingFace model name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Evaluate baseline only"
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Run GPTQ quantization"
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[2, 3, 4], help="Quantization bits"
    )
    parser.add_argument(
        "--naive", action="store_true", help="Use naive RTN instead of GPTQ"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Log to Weights & Biases"
    )
    parser.add_argument(
        "--stride", type=int, default=512, help="Stride for perplexity evaluation"
    )
    parser.add_argument(
        "--n-samples", type=int, default=128, help="Number of calibration samples"
    )
    parser.add_argument(
        "--block-size", type=int, default=128, help="GPTQ column block size"
    )
    parser.add_argument(
        "--token", type=str, default=None, help="HuggingFace API token (for gated models)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Calibration sequence length (default: min(model_max, 2048))",
    )
    parser.add_argument(
        "--group-size", type=int, default=-1,
        help="Grouping: columns per quantization group (-1 = per-row scales, 128 recommended)",
    )
    parser.add_argument(
        "--act-order", action="store_true",
        help="Act-order: quantize columns sorted by descending activation magnitude",
    )
    parser.add_argument(
        "--true-sequential", action="store_true",
        help="True-sequential: quantize sub-layers sequentially within each block",
    )
    parser.add_argument(
        "--all-tricks", action="store_true",
        help="Enable all optimizations (group-size=128, act-order, true-sequential)",
    )
    parser.add_argument(
        "--calib-dataset", type=str, default="wikitext2",
        choices=["wikitext2", "c4"],
        help="Calibration dataset (wikitext2 or c4, default: wikitext2)",
    )
    return parser.parse_args()


def init_wandb(args, n_params):
    """Initialize W&B run with config."""
    import wandb

    method = "rtn" if args.naive else "gptq"
    if not args.quantize:
        method = "baseline"

    run_name = f"{args.model}-{method}"
    if args.quantize:
        run_name += f"-{args.bits}bit"
        if args.group_size > 0:
            run_name += f"-g{args.group_size}"
        if args.act_order:
            run_name += "-actord"
        if args.true_sequential:
            run_name += "-trueseq"

    wandb.init(
        project="gptq-from-scratch",
        name=run_name,
        config={
            "model": args.model,
            "method": method,
            "bits": args.bits if args.quantize else 32,
            "n_params": n_params,
            "device": args.device,
            "stride": args.stride,
            "n_samples": args.n_samples,
            "block_size": args.block_size,
            "group_size": args.group_size,
            "act_order": args.act_order,
            "true_sequential": args.true_sequential,
            "calib_dataset": args.calib_dataset,
        },
    )
    return wandb


def main():
    args = parse_args()

    if args.all_tricks:
        args.group_size = 128
        args.act_order = True
        args.true_sequential = True

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model, device=args.device, token=args.token)
    print(f"Model loaded on {args.device} ({model.dtype})")

    arch = get_arch_config(model)
    max_seq = arch.get_max_seq_len(model)
    print(f"Architecture: {model.config.__class__.__name__}, max_seq_len={max_seq}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    wb = None
    if args.wandb:
        wb = init_wandb(args, n_params)

    if args.baseline or not args.quantize:
        print("\n--- Baseline Evaluation ---")
        t0 = time.time()
        ppl = evaluate_perplexity(model, tokenizer, device=args.device, stride=args.stride)
        elapsed = time.time() - t0
        print(f"WikiText-2 perplexity: {ppl:.2f}")
        print(f"Evaluation time: {elapsed:.1f}s")

        if wb:
            wb.log({"baseline_perplexity": ppl, "baseline_eval_time": elapsed})

    if args.quantize:
        from model_utils import get_calibration_data
        from gptq import quantize_model
        from quantize import round_to_nearest

        seq_len = args.seq_len if args.seq_len else min(max_seq, 2048)
        print(f"\n--- Calibration ({args.calib_dataset}, seq_len={seq_len}) ---")
        calib_data = get_calibration_data(
            tokenizer, n_samples=args.n_samples, seq_len=seq_len,
            dataset_name=args.calib_dataset,
        )
        print(f"Calibration: {len(calib_data)} samples x {seq_len} tokens")

        if args.naive:
            print(f"\n--- Naive RTN Quantization ({args.bits}-bit) ---")
            t0 = time.time()
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.data = round_to_nearest(module.weight.data, args.bits)
                elif type(module).__name__ == "Conv1D":
                    module.weight.data = round_to_nearest(module.weight.data, args.bits)
            quant_time = time.time() - t0
            print(f"RTN done in {quant_time:.1f}s")

            if wb:
                wb.log({"quantization_time": quant_time})
        else:
            tricks = []
            if args.group_size > 0:
                tricks.append(f"g={args.group_size}")
            if args.act_order:
                tricks.append("act-order")
            if args.true_sequential:
                tricks.append("true-seq")
            trick_str = f" [{', '.join(tricks)}]" if tricks else ""
            print(f"\n--- GPTQ Quantization ({args.bits}-bit{trick_str}) ---")
            stats = quantize_model(
                model, calib_data, n_bits=args.bits,
                block_size=args.block_size,
                group_size=args.group_size,
                act_order=args.act_order,
                true_sequential=args.true_sequential,
                device=args.device,
            )
            quant_time = sum(s["time"] for s in stats.values())
            total_loss = sum(s["loss"] for s in stats.values())
            print(f"GPTQ done in {quant_time:.1f}s ({len(stats)} layers)")

            if wb:
                wb.log({
                    "quantization_time": quant_time,
                    "total_gptq_loss": total_loss,
                    "n_layers_quantized": len(stats),
                })
                for layer_name, s in stats.items():
                    wb.log({
                        f"layer_loss/{layer_name}": s["loss"],
                        f"layer_time/{layer_name}": s["time"],
                    })

        print(f"\n--- Post-Quantization Evaluation ---")
        t0 = time.time()
        ppl = evaluate_perplexity(model, tokenizer, device=args.device, stride=args.stride)
        eval_time = time.time() - t0
        print(f"WikiText-2 perplexity ({args.bits}-bit): {ppl:.2f}")
        print(f"Evaluation time: {eval_time:.1f}s")

        if wb:
            wb.log({
                "perplexity": ppl,
                "bits": args.bits,
                "eval_time": eval_time,
            })

    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
