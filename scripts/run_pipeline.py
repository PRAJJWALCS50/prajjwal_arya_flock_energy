"""Run all assignment sections and regenerate the reports from supplied data."""

from pathlib import Path
import argparse, subprocess, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--clean-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.output_dir.resolve()
    raw = a.raw_dir.resolve()
    clean = a.clean_dir.resolve()
    jobs = [
        (
            "raw_house1.py",
            ["--data-dir", str(raw), "--output-dir", str(out / "raw_house1")],
        )
    ]
    jobs += [
        (
            "run_eda.py",
            [
                "--data-dir",
                str(clean),
                "--output-dir",
                str(out / "eda"),
            ],
        ),
        (
            "train_model.py",
            ["--data-dir", str(clean), "--output-dir", str(out / "model")],
        ),
    ]
    for script, argv in jobs:
        print("RUN " + script, flush=True)
        subprocess.run(
            [sys.executable, str(root / "scripts" / script), *argv],
            check=True,
            cwd=root,
        )


if __name__ == "__main__":
    main()
