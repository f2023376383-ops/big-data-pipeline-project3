"""
Project 3 - Step 6: Question 3 - "Which marketing patterns are unusual?"
Finds posts that behave very differently from COMPARABLE posts (same platform and same content type),
so that the marketing team can look at them. A flag means "please investigate", not "this post is bad".

Three signals per post, each compared with the typical post of its own group (robust z-score):
    1. reach signal   : performance index (likes compared with the brand's normal post)  -> viral or flop
    2. chatter signal : comments per like                                                -> unusually talked-about post
    3. anger signal   : share of NEGATIVE comments (binomial z-score, because a post with
                        3 comments and one with 24 comments are not equally reliable)

Reads : data/clean/posts_scored.csv, data/clean/comments_clustered.csv
Writes: data/clean/flagged_posts.csv     (every post with its scores, flag and reason)
        outputs/q3_unusual_posts.png

Run from the project root folder:
    python src/detect_anomalies.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "outputs"

# thresholds: how many "robust standard deviations" away from normal counts as unusual
T_HIGH_REACH, T_LOW_REACH, T_CHATTER, T_ANGER = 2.5, -3.0, 2.5, 3.0

posts = pd.read_csv(CLEAN / "posts_scored.csv")
comments = pd.read_csv(CLEAN / "comments_clustered.csv", keep_default_na=False, na_values=[""])

# ---------------------------------------------------------------- features ----
g = comments.groupby("post_id")
neg = pd.DataFrame({"n_neg": g["sentiment"].apply(lambda s: (s == "Negative").sum()), "n_sampled": g.size()}).reset_index()
# which segment do the negative comments of each post belong to? (this explains WHY the post is angry)
neg_segment = (comments[comments["sentiment"] == "Negative"].groupby("post_id")["segment_name"]
               .agg(lambda s: s.value_counts().index[0]).rename("main_negative_segment").reset_index())

d = posts.merge(neg, on="post_id").merge(neg_segment, on="post_id", how="left")
d["share_negative"] = d["n_neg"] / d["n_sampled"]
d["log_index"] = np.log(d["performance_index"])
d["log_comment_per_like"] = np.log(d["comments_count"] / d["likes"])
d["group"] = d["platform"] + " | " + d["content_type"]

def robust_z(col):
    """(value - group median) / group MAD. MAD is like a standard deviation that big outliers cannot spoil."""
    med = d.groupby("group")[col].transform("median")
    mad = (d[col] - med).abs().groupby(d["group"]).transform("median") * 1.4826
    return (d[col] - med) / mad.clip(lower=mad.median() * 0.5)      # floor so a tiny MAD cannot explode the score

d["z_reach"] = robust_z("log_index")
d["z_chatter"] = robust_z("log_comment_per_like")

# anger: is the share of negative comments much higher than the normal share of this group?
p0 = d.groupby("group")["n_neg"].transform("sum") / d.groupby("group")["n_sampled"].transform("sum")
d["z_anger"] = (d["share_negative"] - p0) / np.sqrt(p0 * (1 - p0) / d["n_sampled"])

# -------------------------------------------------------------------- flags ----
d["flag_high_reach"] = d["z_reach"] > T_HIGH_REACH
d["flag_low_reach"] = d["z_reach"] < T_LOW_REACH
d["flag_chatter"] = d["z_chatter"] > T_CHATTER
d["flag_anger"] = d["z_anger"] > T_ANGER
d["is_unusual"] = d[["flag_high_reach", "flag_low_reach", "flag_chatter", "flag_anger"]].any(axis=1)

def reason(r):
    if not r["is_unusual"]:
        return ""
    if r["flag_anger"]:
        seg = r["main_negative_segment"] if isinstance(r["main_negative_segment"], str) else "negative comments"
        return f"Many negative comments, mostly from: {seg}"
    if r["flag_high_reach"]:
        return "Much more reach than similar posts (possible viral post)"
    if r["flag_low_reach"]:
        return "Much less reach than similar posts (possible flop)"
    return "Many more comments per like than similar posts"
d["reason"] = d.apply(reason, axis=1)
d["strength"] = d[["z_reach", "z_chatter", "z_anger"]].abs().max(axis=1)

flagged = d[d["is_unusual"]].sort_values("strength", ascending=False)

# ------------------------------------------------------------------- report ----
print("=" * 72); print("STEP 6: UNUSUAL POSTS"); print("=" * 72)
print(f"\nPosts checked : {len(d):,}")
print(f"Posts flagged : {len(flagged):,} ({len(flagged) / len(d):.1%})")
print("\nWhy they were flagged:"); print(flagged["reason"].str.split(":").str[0].value_counts().to_string())
print("\nFlagged posts by platform (share of that platform's posts):")
print((flagged["platform"].value_counts() / d["platform"].value_counts()).round(3).to_string())

print("\nTop 10 posts to investigate first:")
show = flagged.head(10)[["post_id", "brand", "platform", "content_type", "campaign", "likes", "comments_count", "reason"]]
print(show.to_string(index=False))

# Because I injected unusual posts into the simulated data on purpose, I can check the detector.
# With real data you have NO such answer key, which is why this check is only a test of the method.
inj = d["injected_anomaly"] != "none"
tp = (d["is_unusual"] & inj).sum()
print("\nCHECK AGAINST THE POSTS I INJECTED ON PURPOSE (possible only because the data is simulated):")
print(f"   Injected unusual posts : {inj.sum()}")
print(f"   Precision (flagged posts that were really injected): {tp / max(len(flagged), 1):.0%}")
print(f"   Recall    (injected posts that were found)         : {tp / inj.sum():.0%}")
print(pd.crosstab(d["injected_anomaly"], d["is_unusual"].map({True: "flagged", False: "not flagged"})).to_string())
print("   Posts with very few comments are hard to judge, so some injected posts are missed. That is normal.")

# --------------------------------------------------------------------- chart ----
fig, ax = plt.subplots(figsize=(8, 5))
normal = d[~d["is_unusual"]]
ax.scatter(normal["z_reach"], normal["z_anger"], s=8, alpha=0.3, color="#999", label="normal posts")
ax.scatter(flagged["z_reach"], flagged["z_anger"], s=16, alpha=0.8, color="#d4423b", label="flagged posts")
ax.axvline(T_HIGH_REACH, color="k", ls=":", lw=0.8); ax.axhline(T_ANGER, color="k", ls=":", lw=0.8)
ax.set_xlabel("Reach score (higher = more likes than similar posts)")
ax.set_ylabel("Anger score (higher = more negative comments than similar posts)")
ax.set_title("Unusual posts: far from the normal cloud"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "q3_unusual_posts.png", dpi=150); plt.close()

keep = ["post_id", "brand", "platform", "campaign", "content_type", "post_date", "likes", "comments_count",
        "shares", "views", "n_sampled", "share_negative", "z_reach", "z_chatter", "z_anger",
        "is_unusual", "reason", "main_negative_segment", "strength", "injected_anomaly"]
d[keep].round(3).to_csv(CLEAN / "flagged_posts.csv", index=False)
print("\nSaved: data/clean/flagged_posts.csv and outputs/q3_unusual_posts.png")
