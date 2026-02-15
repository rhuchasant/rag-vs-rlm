"""Run robustness tests (seeds, format variants) and compute confidence intervals."""

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def mean_std_ci(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95": 0.0}
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def run_and_get_results_path(cmd: List[str], cwd: Path) -> Tuple[str, Path]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{output[:1200]}")
    match = re.search(r"Results saved to\s+(.+)", output)
    if not match:
        raise RuntimeError(f"Could not find results path in output:\n{output[-1200:]}")
    path = Path(match.group(1).strip())
    return output, path


def run_synthetic_suite(args, repo_root: Path) -> Dict:
    runs = []
    intermediate_paths: List[Path] = []
    seeds = parse_int_list(args.synthetic_seeds)
    variants = parse_str_list(args.synthetic_variants)
    total_runs = len(seeds) * len(variants)
    run_idx = 0

    for seed in seeds:
        for variant in variants:
            run_idx += 1
            print(
                f"[synthetic {run_idx}/{total_runs}] seed={seed} variant={variant} starting...",
                flush=True,
            )
            cmd = [
                sys.executable,
                "scripts/run_synthetic_structured_extraction.py",
                "--docs", str(args.synthetic_docs),
                "--sections", str(args.synthetic_sections),
                "--target-chars", str(args.synthetic_target_chars),
                "--top-k", str(args.synthetic_top_k),
                "--seed", str(seed),
                "--format-variant", variant,
            ]
            _, path = run_and_get_results_path(cmd, repo_root)
            intermediate_paths.append(path)
            print(
                f"[synthetic {run_idx}/{total_runs}] completed -> {path.name}",
                flush=True,
            )
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            summary = data.get("summary", {})
            runs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "results_file": str(path),
                    "rag_overall_exact_rate": summary.get("rag_overall_exact_rate", 0.0),
                    "rag_avg_topic_accuracy_on_matched": summary.get("rag_avg_topic_accuracy_on_matched", 0.0),
                    "rlm_overall_exact_rate": summary.get("rlm_overall_exact_rate", 0.0),
                    "rlm_avg_topic_accuracy_on_matched": summary.get("rlm_avg_topic_accuracy_on_matched", 0.0),
                }
            )

    rag_exact = [r["rag_overall_exact_rate"] for r in runs]
    rag_topic = [r["rag_avg_topic_accuracy_on_matched"] for r in runs]
    rlm_exact = [r["rlm_overall_exact_rate"] for r in runs]
    rlm_topic = [r["rlm_avg_topic_accuracy_on_matched"] for r in runs]

    return {
        "config": {
            "seeds": parse_int_list(args.synthetic_seeds),
            "variants": parse_str_list(args.synthetic_variants),
            "docs": args.synthetic_docs,
            "sections": args.synthetic_sections,
            "target_chars": args.synthetic_target_chars,
            "top_k": args.synthetic_top_k,
        },
        "aggregate": {
            "rag_overall_exact_rate": mean_std_ci(rag_exact),
            "rag_topic_accuracy": mean_std_ci(rag_topic),
            "rlm_overall_exact_rate": mean_std_ci(rlm_exact),
            "rlm_topic_accuracy": mean_std_ci(rlm_topic),
        },
        "runs": runs,
        "intermediate_files": [str(p) for p in intermediate_paths],
    }


def run_excel_suite(args, repo_root: Path) -> Dict:
    runs = []
    intermediate_paths: List[Path] = []
    seeds = parse_int_list(args.excel_seeds)
    total_runs = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        print(f"[excel {idx}/{total_runs}] seed={seed} starting...", flush=True)
        cmd = [
            sys.executable,
            "scripts/run_comparison.py",
            "--generate",
            "--tabs", str(args.excel_tabs),
            "--size", str(args.excel_size),
            "--top-k", str(args.excel_top_k),
            "--seed", str(seed),
        ]
        _, path = run_and_get_results_path(cmd, repo_root)
        intermediate_paths.append(path)
        print(f"[excel {idx}/{total_runs}] completed -> {path.name}", flush=True)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        num_tabs = data.get("num_tabs", 19)
        rag_cfg = data.get("rag_configs", {})
        top5_cov = rag_cfg.get("top_k=5_semantic", {}).get("coverage")
        top_sem_key = f"top_k={num_tabs}_semantic"
        top_tab_key = f"top_k={num_tabs}_tab_chunking"
        top_sem_err = "error" in rag_cfg.get(top_sem_key, {})
        top_tab_err = "error" in rag_cfg.get(top_tab_key, {})
        rlm_cov = data.get("rlm", {}).get("coverage")
        runs.append(
            {
                "seed": seed,
                "results_file": str(path),
                "rag_top5_coverage": float(top5_cov) if top5_cov is not None else 0.0,
                "rag_full_semantic_context_error": bool(top_sem_err),
                "rag_full_tab_context_error": bool(top_tab_err),
                "rlm_coverage": float(rlm_cov) if rlm_cov is not None else 0.0,
            }
        )

    top5_covs = [r["rag_top5_coverage"] for r in runs]
    rlm_covs = [r["rlm_coverage"] for r in runs]
    sem_err_rate = sum(1 for r in runs if r["rag_full_semantic_context_error"]) / len(runs) if runs else 0.0
    tab_err_rate = sum(1 for r in runs if r["rag_full_tab_context_error"]) / len(runs) if runs else 0.0

    return {
        "config": {
            "seeds": parse_int_list(args.excel_seeds),
            "tabs": args.excel_tabs,
            "size": args.excel_size,
            "top_k": args.excel_top_k,
        },
        "aggregate": {
            "rag_top5_coverage": mean_std_ci(top5_covs),
            "rlm_coverage": mean_std_ci(rlm_covs),
            "rag_full_semantic_context_error_rate": sem_err_rate,
            "rag_full_tab_context_error_rate": tab_err_rate,
        },
        "runs": runs,
        "intermediate_files": [str(p) for p in intermediate_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-excel", action="store_true")
    parser.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help="Delete per-run intermediate result JSONs after writing final robustness report",
    )

    parser.add_argument("--synthetic-seeds", default="41,42,43")
    parser.add_argument("--synthetic-variants", default="standard,alt_headers,noisy,mixed_case")
    parser.add_argument("--synthetic-docs", type=int, default=5)
    parser.add_argument("--synthetic-sections", type=int, default=20)
    parser.add_argument("--synthetic-target-chars", type=int, default=50000)
    parser.add_argument("--synthetic-top-k", type=int, default=20)

    parser.add_argument("--excel-seeds", default="42,43,44")
    parser.add_argument("--excel-tabs", type=int, default=30)
    parser.add_argument("--excel-size", type=int, default=750000)
    parser.add_argument("--excel-top-k", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    out = {"timestamp": datetime.now().isoformat()}

    if not args.skip_synthetic:
        print("Running synthetic robustness suite...")
        out["synthetic"] = run_synthetic_suite(args, repo_root)
    if not args.skip_excel:
        print("Running excel robustness suite...")
        out["excel"] = run_excel_suite(args, repo_root)

    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"robustness_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if args.cleanup_intermediate:
        deleted = 0
        for key in ["synthetic", "excel"]:
            if key in out:
                for p_str in out[key].get("intermediate_files", []):
                    p = Path(p_str)
                    try:
                        if p.exists() and p.resolve() != out_path.resolve():
                            p.unlink()
                            deleted += 1
                    except Exception:
                        pass
        print(f"Deleted {deleted} intermediate result files.", flush=True)

    print(f"\nRobustness report saved to {out_path}")
    if "synthetic" in out:
        s = out["synthetic"]["aggregate"]
        print(
            "Synthetic | RAG exact "
            f"{s['rag_overall_exact_rate']['mean']:.3f}+/-{s['rag_overall_exact_rate']['ci95']:.3f} | "
            "RLM exact "
            f"{s['rlm_overall_exact_rate']['mean']:.3f}+/-{s['rlm_overall_exact_rate']['ci95']:.3f}"
        )
    if "excel" in out:
        e = out["excel"]["aggregate"]
        print(
            "Excel | RAG top5 cov "
            f"{e['rag_top5_coverage']['mean']:.3f}+/-{e['rag_top5_coverage']['ci95']:.3f} | "
            "RLM cov "
            f"{e['rlm_coverage']['mean']:.3f}+/-{e['rlm_coverage']['ci95']:.3f}"
        )


if __name__ == "__main__":
    main()
