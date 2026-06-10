import argparse
import csv
import os
from pathlib import Path


VARIANTS = {
    "vanilla": "",
    "full_beta": (
        "--use-conf-scores --confidence-mode beta "
        "--lambda-sparsity 0.01 --beta-ent 0.002 --rank-weight 1.0"
    ),
    "no_sparsity": (
        "--use-conf-scores --confidence-mode beta "
        "--lambda-sparsity 0.0 --beta-ent 0.002 --rank-weight 1.0"
    ),
    "no_entropy": (
        "--use-conf-scores --confidence-mode beta "
        "--lambda-sparsity 0.01 --beta-ent 0.0 --rank-weight 1.0"
    ),
    "no_ranking": (
        "--use-conf-scores --confidence-mode beta "
        "--lambda-sparsity 0.01 --beta-ent 0.002 --rank-weight 0.0"
    ),
    "scalar_conf": (
        "--use-conf-scores --confidence-mode scalar "
        "--lambda-sparsity 0.01 --beta-ent 0.0 --rank-weight 1.0"
    ),
}


DEFAULT_SCENES = [
    "BigBen_Scene",
    "Eiffel_Tower_Scene",
    "Louvre_Museum_Scene",
    "Perspolis_Scene",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a TSV manifest for review-response Slurm arrays."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(f"/scratch/{os.environ.get('USER', 'USER')}/datasets/Confident_Splatting"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            f"/scratch/{os.environ.get('USER', 'USER')}/results/confident_splatting_review"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/review_manifest.tsv"))
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--configs", nargs="+", default=["default"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=sorted(VARIANTS.keys()),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write rows even when a scene directory does not exist yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in args.scenes:
        data_dir = args.data_root / scene
        if not data_dir.exists() and not args.allow_missing:
            print(f"Skipping missing scene: {data_dir}")
            continue
        for config in args.configs:
            for variant_name in args.variants:
                variant = f"{config}_{variant_name}"
                result_dir = args.results_root / scene / variant
                rows.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "config": config,
                        "data_dir": str(data_dir),
                        "result_dir": str(result_dir),
                        "extra_args": VARIANTS[variant_name],
                    }
                )

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "variant", "config", "data_dir", "result_dir", "extra_args"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
