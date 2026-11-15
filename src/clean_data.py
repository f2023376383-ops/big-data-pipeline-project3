"""
Project 3 - Step 3: Data quality checks and cleaning
(same habit as Project 1: look first, then clean a COPY. The raw files are never changed.)

Reads : data/raw/posts.csv, data/raw/comments.csv
Writes: data/clean/posts_clean.csv, data/clean/comments_clean.csv
        outputs/data_quality_report.txt   (everything this script prints, for your documentation)

Run from the project root folder:
    python src/clean_data.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN, OUT = ROOT / "data" / "raw", ROOT / "data" / "clean", ROOT / "outputs"
CLEAN.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

report = []
def say(text=""):
    print(text)
    report.append(str(text))


# ----------------------------------------------------------------------------
# Roman Urdu has no fixed spelling: "bohat", "bohot", "boht" are the same word.
# We map the common variants to ONE spelling so that clustering treats them as the same word.
# ----------------------------------------------------------------------------
SPELLING_MAP = {
    r"\b(bohat|bohot|boht|bhot|bahut)\b": "bohat",
    r"\b(zyada|ziada|zyadah|ziyada)\b": "zyada",
    r"\b(kitne|kitnay|kitnai|kitney)\b": "kitne",
    r"\b(kya|kia|kiya)\b": "kya",
    r"\b(ha|hai|hay)\b": "hai",
    r"\b(nahi|nahin|nhi|nai)\b": "nahi",
    r"\b(mehnga|mehngi|mehanga|mahnga|mehnge)\b": "mehnga",
    r"\b(qeemat|qimat|keemat)\b": "qeemat",
    r"\b(aya|aaya)\b": "aya",
    r"\b(kesi|kaisi|kesa|kaisa)\b": "kaisi",
    r"\b(jesa|jaisa)\b": "jaisa",
    r"\b(kpra|kapra|kapda)\b": "kapra",
    r"\b(milega|milay ga|milay gaa)\b": "milega",
}

def clean_text(text):
    """lowercase, remove emojis and punctuation, fix common Roman Urdu spelling variants."""
    t = str(text).lower()
    t = re.sub(r"[^\w\s']", " ", t)          # emojis and punctuation become spaces (Urdu letters are kept)
    for pattern, replacement in SPELLING_MAP.items():
        t = re.sub(pattern, replacement, t)
    return re.sub(r"\s+", " ", t).strip()

URDU_SCRIPT = re.compile(r"[\u0600-\u06FF]")


# ============================================================================
say("=" * 70)
say("STEP 3: DATA QUALITY REPORT (simulated data)")
say("=" * 70)

# Only EMPTY cells count as missing. (By default pandas also treats the word "None" as missing,
# but in our comments "None" is a real answer: "this comment shows no purchase intent".)
posts_raw = pd.read_csv(RAW / "posts.csv", keep_default_na=False, na_values=[""])
comments_raw = pd.read_csv(RAW / "comments.csv", keep_default_na=False, na_values=[""])

# ---------------------------------------------------------------- CHECKS ----
say("\n--- A. Posts table: basic checks ---")
say(f"Rows, columns: {posts_raw.shape}")
say("Missing values per column:")
say(posts_raw.isna().sum()[lambda s: s > 0].to_string() or "none")
say(f"Fully duplicated rows : {posts_raw.duplicated().sum()}")
say(f"Duplicate post_id     : {posts_raw['post_id'].duplicated().sum()}")
for col in ["likes", "comments_count", "shares", "views"]:
    say(f"Negative {col:<15}: {(posts_raw[col] < 0).sum()}")
say(f"likes > views (not possible): {(posts_raw['likes'] > posts_raw['views']).sum()}")
say(f"Date range of posts: {posts_raw['post_date'].min()} to {posts_raw['post_date'].max()}")
say("\nPosts per platform:");      say(posts_raw["platform"].value_counts().to_string())
say("\nPosts per campaign:");      say(posts_raw["campaign"].value_counts().to_string())
say("\nPosts per content type:");  say(posts_raw["content_type"].value_counts().to_string())
say(f"\nViews missing: {posts_raw['views'].isna().sum()} posts "
    f"({posts_raw['views'].isna().mean():.1%}). Only Facebook has missing views:")
say(posts_raw[posts_raw["views"].isna()]["platform"].value_counts().to_string())

say("\n--- B. Comments table: basic checks ---")
say(f"Rows, columns: {comments_raw.shape}")
say("Missing values per column:")
say(comments_raw.isna().sum()[lambda s: s > 0].to_string() or "none")
say(f"Comments whose post_id is not in posts (orphans): {(~comments_raw['post_id'].isin(posts_raw['post_id'])).sum()}")
say(f"Posts that have no comment in the sample       : {(~posts_raw['post_id'].isin(comments_raw['post_id'])).sum()}")
say(f"Empty comment_text                              : {(comments_raw['comment_text'].astype(str).str.strip() == '').sum()}")
n_dup_texts = comments_raw.duplicated(subset=["comment_text"]).sum()
say(f"Repeated comment_text values: {n_dup_texts:,} "
    f"({n_dup_texts / len(comments_raw):.0%}). We KEEP them: many different people really write 'Price?'.")
say("   Same wording from many people is a signal (demand), not a data error.")
per_post = comments_raw.groupby("post_id").size()
check = posts_raw.set_index("post_id")["comments_count"].reindex(per_post.index)
say(f"Posts where sampled comments > comments_count: {(per_post > check).sum()}")
say("\nComments per language:");  say(comments_raw["language_type"].value_counts().to_string())
say("\nComments per sentiment:"); say(comments_raw["sentiment"].value_counts().to_string())
say("\nComments per purchase_intent:"); say(comments_raw["purchase_intent"].value_counts().to_string())
urdu_mismatch = (comments_raw["comment_text"].astype(str).apply(lambda s: bool(URDU_SCRIPT.search(s)))
                 != (comments_raw["language_type"] == "Urdu")).sum()
say(f"\nUrdu-script text labelled as another language (or the reverse): {urdu_mismatch}")

# -------------------------------------------------------------- CLEANING ----
say("\n" + "=" * 70)
say("CLEANING (on copies; the raw files stay unchanged)")
say("=" * 70)

posts = posts_raw.drop_duplicates(subset="post_id").copy()
for col in ["platform", "content_type", "brand", "campaign"]:
    posts[col] = posts[col].astype(str).str.strip()
posts["post_date"] = pd.to_datetime(posts["post_date"])
for col in ["likes", "comments_count", "shares", "views"]:
    posts[col] = pd.to_numeric(posts[col], errors="coerce")
    posts.loc[posts[col] < 0, col] = np.nan          # negative counts are impossible -> blank (P1 rule)
posts.loc[posts["likes"] > posts["views"], "views"] = np.nan   # impossible -> blank
# Views stay blank when missing. We do NOT fill them with 0 or the average ("not collected" is not "zero").

posts["post_month"] = posts["post_date"].dt.to_period("M").astype(str)
posts["total_engagement"] = posts[["likes", "comments_count", "shares"]].sum(axis=1)
posts["engagement_rate"] = posts["total_engagement"] / posts["views"]     # NaN when views are missing
posts["injected_anomaly"] = posts["injected_anomaly"].fillna("none")

comments = comments_raw.dropna(subset=["comment_text"]).copy()
comments = comments[comments["post_id"].isin(posts["post_id"])]
comments["comment_text"] = comments["comment_text"].astype(str).str.strip()
comments["purchase_intent"] = comments["purchase_intent"].replace("None", "No intent")
comments["text_clean"] = comments["comment_text"].apply(clean_text)
comments["has_urdu_script"] = comments["comment_text"].apply(lambda s: bool(URDU_SCRIPT.search(s)))
comments["word_count"] = comments["text_clean"].str.split().str.len()
comments = comments[comments["text_clean"] != ""]

say(f"Posts   : {len(posts_raw):,} -> {len(posts):,} rows")
say(f"Comments: {len(comments_raw):,} -> {len(comments):,} rows")
say(f"New post columns   : post_month, total_engagement, engagement_rate")
say(f"New comment columns: text_clean, has_urdu_script, word_count")
say(f"Engagement rate can be calculated for {posts['engagement_rate'].notna().sum():,} of {len(posts):,} posts "
    f"(the rest have no views).")

say("\nExamples of text cleaning (before -> after):")
for s in comments.sample(6, random_state=1).itertuples():
    say(f"   {s.comment_text!r:45} -> {s.text_clean!r}")

say("\nMedian likes by content type (clean data):")
say(posts.groupby("content_type")["likes"].median().sort_values(ascending=False).round(0).to_string())

posts.to_csv(CLEAN / "posts_clean.csv", index=False)
comments.to_csv(CLEAN / "comments_clean.csv", index=False, encoding="utf-8-sig")
say("\nSaved: data/clean/posts_clean.csv and data/clean/comments_clean.csv")

(OUT / "data_quality_report.txt").write_text("\n".join(report), encoding="utf-8")
print("Saved: outputs/data_quality_report.txt")
