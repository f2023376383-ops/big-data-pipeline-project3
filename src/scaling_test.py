"""
Project 3 - Step 10 (bonus): Scaling test - what happens when there are many more comments?
Same idea as Project 2 (Pandas vs Spark): copy the comments many times and measure the time.

IMPORTANT: copying comments does not create new information. This test measures SPEED and the
limits of the tools, not the quality of the clusters.

Two groups of tools are timed on 1x, 5x, 10x, 20x ... copies of the 35,195 comments:
    A) Pandas + scikit-learn  : group-by counting, TF-IDF, K-Means, MiniBatch K-Means
    B) Spark (local mode)     : group-by counting, TF-IDF + K-Means with Spark MLlib
       (only if Java and pyspark are installed; otherwise this part is skipped)

Reads : data/clean/comments_clustered.csv, data/clean/posts_scored.csv
Writes: data/clean/scaling_results.csv, outputs/scaling_times.png

Run from the project root folder:
    python src/scaling_test.py                 (copies: 1, 5, 10, 20)
    python src/scaling_test.py --factors 1 5   (smaller test, if your computer is slow)
"""
from pathlib import Path
import argparse
import shutil
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "outputs"
SEED, K = 42, 10

parser = argparse.ArgumentParser()
parser.add_argument("--factors", type=int, nargs="+", default=[1, 5, 10, 20])
parser.add_argument("--time-limit", type=float, default=300,
                    help="stop making the data bigger when one step needs more than this many seconds")
args = parser.parse_args()

comments = pd.read_csv(CLEAN / "comments_clustered.csv", keep_default_na=False, na_values=[""])
posts = pd.read_csv(CLEAN / "posts_scored.csv", usecols=["post_id", "platform", "campaign"])
base = comments.merge(posts, on="post_id")[["text_clean", "platform", "campaign", "segment_name"]]
base["text_clean"] = base["text_clean"].fillna("")

results = []     # one dict per measurement

def timed(label, tool, rows, fn):
    t = time.time()
    out = fn()
    sec = time.time() - t
    results.append({"tool": tool, "step": label, "rows": rows, "seconds": round(sec, 2)})
    print(f"   {tool:<22} {label:<26} {rows:>9,} rows  {sec:>7.1f} s")
    return out, sec


# ============================================================================ A) Pandas + scikit-learn
print("=" * 78); print("A) PANDAS + SCIKIT-LEARN"); print("=" * 78)
stopped_at = None
for f in args.factors:
    df = pd.concat([base] * f, ignore_index=True)
    n = len(df)
    print(f"\n{f}x copies = {n:,} comments")
    timed("group-by counts", "pandas", n, lambda: df.groupby(["platform", "campaign", "segment_name"]).size())
    (vec, X), t_feat = timed("TF-IDF features", "scikit-learn", n, lambda: (lambda v: (v, v.fit_transform(df["text_clean"])))(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)))
    _, t_km = timed("K-Means (full)", "scikit-learn", n, lambda: KMeans(K, n_init=1, random_state=SEED).fit(X))
    _, t_mb = timed("MiniBatch K-Means", "scikit-learn", n, lambda: MiniBatchKMeans(K, n_init=1, batch_size=4096, random_state=SEED).fit(X))
    if max(t_feat, t_km) > args.time_limit:
        stopped_at = f
        print(f"   One step took longer than {args.time_limit:.0f} s, so the bigger copies are skipped.")
        break
    del df, X

# ============================================================================ B) Spark
print("\n" + "=" * 78); print("B) SPARK (local mode on this computer)"); print("=" * 78)
spark_ok = False
if shutil.which("java") is None:
    print("Java was not found, so the Spark part is skipped. (To use Spark: install Java 17 and restart cmd.)")
else:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import Tokenizer, HashingTF, IDF
        from pyspark.ml.clustering import KMeans as SparkKMeans
        spark_ok = True
    except ImportError:
        print("pyspark is not installed, so the Spark part is skipped. (pip install pyspark)")

if spark_ok:
    import os
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)        # avoids 'Python was not found' errors on Windows
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder.master("local[*]").appName("p3-scaling")
             .config("spark.driver.memory", "2g").config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    sdf0 = spark.createDataFrame(base).cache()
    sdf0.count()                                                    # load once, outside the timing
    for f in args.factors:
        if stopped_at is not None and f > stopped_at:
            break
        sdf = sdf0.crossJoin(spark.range(f)).drop("id").repartition(8).cache()      # copy every row f times, inside Spark
        n = sdf.count()
        print(f"\n{f}x copies = {n:,} comments")
        timed("group-by counts", "spark", n, lambda: sdf.groupBy("platform", "campaign", "segment_name").count().collect())
        pipe = Pipeline(stages=[Tokenizer(inputCol="text_clean", outputCol="words"),
                                HashingTF(inputCol="words", outputCol="tf", numFeatures=4096),
                                IDF(inputCol="tf", outputCol="features", minDocFreq=3)])
        (feat, t_feat) = timed("TF-IDF features", "spark MLlib", n, lambda: (lambda m: m.transform(sdf).cache())(pipe.fit(sdf)))
        feat.count()
        timed("K-Means", "spark MLlib", n, lambda: SparkKMeans(k=K, seed=SEED, maxIter=20, featuresCol="features").fit(feat))
        sdf.unpersist(); feat.unpersist()
    spark.stop()

# ============================================================================ results
res = pd.DataFrame(results)
res.to_csv(CLEAN / "scaling_results.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, steps, title in [(axes[0], ["group-by counts"], "Counting (group-by)"),
                         (axes[1], ["K-Means (full)", "MiniBatch K-Means", "K-Means"], "K-Means clustering")]:
    for (tool, step), g in res[res["step"].isin(steps)].groupby(["tool", "step"]):
        ax.plot(g["rows"], g["seconds"], marker="o", label=f"{tool}: {step}")
    ax.set_title(title); ax.set_xlabel("number of comments"); ax.set_ylabel("seconds"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "scaling_times.png", dpi=150); plt.close()

print("\n" + "=" * 78); print("SUMMARY (seconds)"); print("=" * 78)
print(res.pivot_table(index=["tool", "step"], columns="rows", values="seconds").round(1).to_string())
print("\nHow much slower did each step get? (biggest copy compared with the 1x copy)")
lo, hi = res["rows"].min(), res["rows"].max()
print(f"   Data grew {hi / lo:.0f}x ({lo:,} -> {hi:,} comments).")
for (tool, step), g in res.groupby(["tool", "step"]):
    t_lo, t_hi = g.loc[g["rows"].idxmin(), "seconds"], g.loc[g["rows"].idxmax(), "seconds"]
    if t_lo >= 0.2:
        print(f"   {tool:<14} {step:<20} {t_lo:>6.1f} s -> {t_hi:>6.1f} s  (x{t_hi / t_lo:.1f})")
    else:
        print(f"   {tool:<14} {step:<20} {t_lo:>6.1f} s -> {t_hi:>6.1f} s  (too fast at 1x to give a ratio)")
print("\nHow to read this:")
print(" - The first Spark numbers include start-up and warm-up time, so Spark looks slow on the small copy.")
print(" - On ONE computer, pandas/scikit-learn is often faster than Spark for data of this size.")
print(" - Spark is built for data that no longer fits in the memory of one computer, or for many machines.")
print("   This test cannot show that, because Spark ran in local mode here.")
print(" - Copied comments carry no new information, so this measures speed only, not cluster quality.")
print("Saved: data/clean/scaling_results.csv and outputs/scaling_times.png")
