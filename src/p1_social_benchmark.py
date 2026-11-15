"""
Pipeline stage "Project 1": organic social-media benchmark.

This is the cleaning and analysis logic of the Project 1 notebook, turned into a function so the pipeline
can run it. It uses the SAME public dataset as Project 1 (mrudukp/social-media-sentiment-analysis on GitHub,
2,260 posts, NOT from Pakistani brands). The result is only an outside benchmark: how do platforms compare
for organic (unpaid) posts?

Writes: data/raw/social_media_sentiment_raw.csv   (downloaded copy of the raw file, never edited)
        data/clean/p1_platform_benchmark.csv      (median likes, comments, shares per platform)
        data/clean/p1_sentiment_benchmark.csv     (median likes per sentiment)

Run from the project root folder:
    python src/p1_social_benchmark.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR, CLEAN_DIR = ROOT / "data" / "raw", ROOT / "data" / "clean"
URL = "https://raw.githubusercontent.com/mrudukp/social-media-sentiment-analysis/main/social_media_sentiment_raw.csv"


def load_raw():
    """Use the saved copy if it exists, otherwise download it once and save it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / "social_media_sentiment_raw.csv"
    if path.exists():
        return pd.read_csv(path)
    raw = pd.read_csv(URL)               # needs internet. If it fails, download the CSV by hand and save it at `path`
    raw.to_csv(path, index=False)
    return raw


def clean(raw):
    """Same cleaning as the Project 1 notebook (Step 5). The raw data stays unchanged; we work on a copy."""
    df = raw.drop_duplicates().copy()
    df["platform"] = df["platform"].str.strip().str.title()
    df["followers_count"] = pd.to_numeric(df["followers_count"].astype(str).str.replace(",", ""), errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.loc[df["likes"] < 0, "likes"] = np.nan          # negative likes are impossible -> blank, not deleted
    df["hashtag_count"] = df["hashtags"].fillna("").apply(lambda s: len(s.split()) if s else 0)
    return df


def run():
    raw = load_raw()
    df = clean(raw)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    by_platform = (df.groupby("platform").agg(posts=("likes", "size"), median_likes=("likes", "median"),
                                              median_comments=("comments", "median"),
                                              median_shares=("retweets_shares", "median"))
                   .sort_values("median_likes", ascending=False).reset_index())
    by_sentiment = (df.groupby("sentiment").agg(posts=("likes", "size"), median_likes=("likes", "median"))
                    .sort_values("median_likes", ascending=False).reset_index())
    by_platform.to_csv(CLEAN_DIR / "p1_platform_benchmark.csv", index=False)
    by_sentiment.to_csv(CLEAN_DIR / "p1_sentiment_benchmark.csv", index=False)

    print("=" * 72); print("STAGE P1: ORGANIC SOCIAL-MEDIA BENCHMARK (public dataset, not Pakistani)"); print("=" * 72)
    print(f"Rows: {len(raw):,} raw -> {len(df):,} after removing repeated rows")
    print("\nMedian likes per platform:"); print(by_platform.to_string(index=False))
    print("\nMedian likes per sentiment:"); print(by_sentiment.to_string(index=False))
    print("\nSaved: data/clean/p1_platform_benchmark.csv, data/clean/p1_sentiment_benchmark.csv")
    return by_platform, by_sentiment


if __name__ == "__main__":
    run()
