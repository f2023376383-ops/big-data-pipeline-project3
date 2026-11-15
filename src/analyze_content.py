"""
Project 3 - Step 4: Question 1 - "Which content attracts attention?"
(Same method as Project 1: median instead of mean, show how many posts each number is based on.)

Reads : data/clean/posts_clean.csv
Writes: data/clean/posts_scored.csv          (posts + performance_index, used again in later steps)
        data/clean/content_performance.csv   (summary table, used by the final pipeline and the web app)
        outputs/q1_content_type.png, outputs/q1_heatmap.png, outputs/q1_campaign.png

Run from the project root folder:
    python src/analyze_content.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # save charts as files (no window needed)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

posts = pd.read_csv(CLEAN / "posts_clean.csv")

# ---------------------------------------------------------------------------
# Why a "performance index"?
# A big brand gets more likes than a small brand for ANY post, and TikTok gets different numbers
# than Facebook. If we compared raw likes, we would only learn "which brand is big".
# So each post is compared with the typical post of the SAME brand on the SAME platform:
#     performance_index = likes / median likes of that brand + platform
# 1.0 = normal for that brand. 1.5 = 50% better than normal. 0.7 = 30% worse than normal.
# ---------------------------------------------------------------------------
typical = posts.groupby(["brand", "platform"])["likes"].transform("median")
posts["performance_index"] = posts["likes"] / typical

def summarize(by):
    g = posts.groupby(by)
    out = pd.DataFrame({
        "n_posts": g.size(),
        "median_likes": g["likes"].median(),
        "median_index": g["performance_index"].median(),
        "median_comments": g["comments_count"].median(),
        "median_shares": g["shares"].median(),
        "median_engagement_rate": g["engagement_rate"].median(),   # only posts that have views
        "posts_with_views": g["views"].count(),
    })
    return out.round(3)

by_content = summarize("content_type").sort_values("median_index", ascending=False)
by_campaign = summarize("campaign").sort_values("median_index", ascending=False)
by_platform = summarize("platform").sort_values("median_index", ascending=False)
by_cp = summarize(["content_type", "platform"]).reset_index()

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("=" * 70); print("Q1: WHICH CONTENT ATTRACTS ATTENTION?"); print("=" * 70)
print("\nBy content type (sorted by performance index):\n", by_content.to_string())
print("\nBy campaign:\n", by_campaign.to_string())
print("\nBy platform (index is ~1.0 for every platform by design, so look at median_likes):\n", by_platform.to_string())

# ---------------------------------------------------------------- charts ----
# Chart 1: performance index by content type
fig, ax = plt.subplots(figsize=(8, 4.3))
bars = ax.bar(by_content.index, by_content["median_index"], color="#3b6fd4")
ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="normal post of the same brand (1.0)")
for b, n in zip(bars, by_content["n_posts"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={n}", ha="center", fontsize=9)
ax.set_title("Median performance index by content type"); ax.set_ylabel("Performance index")
ax.legend(); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
plt.savefig(OUT / "q1_content_type.png", dpi=150); plt.close()

# Chart 2: heatmap content type x platform
pivot = by_cp.pivot(index="content_type", columns="platform", values="median_index").loc[by_content.index]
counts = by_cp.pivot(index="content_type", columns="platform", values="n_posts").loc[by_content.index]
fig, ax = plt.subplots(figsize=(7, 4.3))
im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax.text(j, i, f"{pivot.values[i, j]:.2f}\n(n={int(counts.values[i, j])})", ha="center", va="center", fontsize=8)
ax.set_title("Performance index: content type x platform"); plt.colorbar(im)
plt.tight_layout(); plt.savefig(OUT / "q1_heatmap.png", dpi=150); plt.close()

# Chart 3: campaign
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(by_campaign.index, by_campaign["median_index"], color="#d4863b")
ax.axhline(1.0, color="k", linestyle="--", linewidth=1)
for b, n in zip(bars, by_campaign["n_posts"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={n}", ha="center", fontsize=9)
ax.set_title("Median performance index by campaign"); ax.set_ylabel("Performance index")
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(OUT / "q1_campaign.png", dpi=150); plt.close()

# ----------------------------------------------------------------- save -----
posts.to_csv(CLEAN / "posts_scored.csv", index=False)
by_cp.to_csv(CLEAN / "content_performance.csv", index=False)

best, worst = by_content.index[0], by_content.index[-1]
print("\n" + "-" * 70)
print(f"Best content type : {best} (index {by_content.loc[best, 'median_index']})")
print(f"Weakest           : {worst} (index {by_content.loc[worst, 'median_index']})")
small = by_cp[by_cp["n_posts"] < 30]
print(f"Groups with fewer than 30 posts (be careful with these): {len(small)} of {len(by_cp)}")
print("\nSaved: data/clean/posts_scored.csv, data/clean/content_performance.csv")
print("Saved charts in outputs/: q1_content_type.png, q1_heatmap.png, q1_campaign.png")
