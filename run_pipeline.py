"""
Run the whole pipeline (Project 1 -> Project 2 -> Project 3 -> recommendations) with ONE command.

    python run_pipeline.py                  (runs every stage, creates new data)
    python run_pipeline.py --keep-data      (keeps the existing simulated data in data/raw, runs the rest)

Every stage is a script in src/. If one stage fails, the pipeline stops and tells you which one.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("P3  generate simulated posts and comments", "generate_data.py"),
    ("P3  quality checks and cleaning",           "clean_data.py"),
    ("P1  organic benchmark (public dataset)",    "p1_social_benchmark.py"),
    ("P2  ad performance and budget",             "p2_budget.py"),
    ("P3  Q1 content performance",                "analyze_content.py"),
    ("P3  Q2/Q4 comment clustering (audience)",   "cluster_comments.py"),
    ("P3  Q3 unusual posts",                      "detect_anomalies.py"),
    ("ALL Q5 recommendations (P1 + P2 + P3)",     "build_recommendations.py"),
]

if __name__ == "__main__":
    keep = "--keep-data" in sys.argv
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t_all = time.time()
    for i, (label, script) in enumerate(STAGES, start=1):
        if keep and script == "generate_data.py":
            print(f"[{i}/{len(STAGES)}] {label}: skipped (--keep-data)")
            continue
        print(f"\n[{i}/{len(STAGES)}] {label}  ({script})")
        t = time.time()
        result = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(result.stdout[-1500:]); print(result.stderr[-2500:])
            print(f"\nSTOPPED: stage '{script}' failed. Fix it and run again.")
            sys.exit(1)
        last = [l for l in result.stdout.strip().splitlines() if l.strip()][-1:]
        print(f"      done in {time.time() - t:.1f}s   {last[0] if last else ''}")
    print(f"\nPipeline finished in {time.time() - t_all:.0f}s. Results are in data/clean/ and outputs/.")
