"""
Project 3 - Step 5: Questions 2 and 4 - "What does the audience want?" / "Which audience needs should the brand address?"
Groups similar comments with K-Means clustering (unsupervised: the algorithm does NOT see the topic column).

Reads : data/clean/comments_clean.csv, data/clean/posts_scored.csv
Writes: data/clean/comments_clustered.csv           (every comment with its cluster)
        data/clean/cluster_summary.csv              (one row per audience segment)
        data/clean/segment_share_by_platform_campaign.csv   (link to Project 2: platform x campaign)
        data/clean/post_segment_features.csv        (share of each segment per post, used in Step 6)
        outputs/q2_segment_sizes.png, outputs/q2_scatter.png, outputs/q2_by_platform.png

Run from the project root folder:
    python src/cluster_comments.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

if hasattr(sys.stdout, "reconfigure"):                 # so Urdu text prints without errors
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

K = 10                  # number of audience segments (see the "choosing k" table printed below)
SEED = 42
CONCEPT_WEIGHT = 10     # how many times a concept token is repeated (3 = too weak and unstable, 30 = just copies my keyword list)

comments = pd.read_csv(CLEAN / "comments_clean.csv", keep_default_na=False, na_values=[""])
posts = pd.read_csv(CLEAN / "posts_scored.csv")

# ---------------------------------------------------------------------------
# Domain word list ("concept lexicon").
# Problem: the same need is written in 3 languages ("Price please?", "Price kya hai?", Urdu script)
# and these share almost no words, so a plain text model splits comments by LANGUAGE, not by NEED.
# Fix: for each comment we add a few extra "concept tokens" (cpt_price_ask, cpt_delivery, ...)
# when it contains one of these keywords. The keywords are my own list, written after reading
# the comments. It is a normal feature-engineering step, but it is human knowledge that I put in,
# so it must be mentioned in the documentation.
# ---------------------------------------------------------------------------
CONCEPTS = {
    "cpt_price_ask":  ["price", "rate", "qeemat", "kitni", "how much", "kitne ka", "قیمت", "کتنے کا"],
    "cpt_price_high": ["expensive", "mehnga", "zyada", "too high", "very high", "going up", "مہنگا", "زیادہ"],
    "cpt_discount":   ["discount", "sale", "offer", "code", "kam ho", "ڈسکاؤنٹ", "سیل"],
    "cpt_delivery":   ["deliver", "delivery", "parcel", "charges", "ڈیلیوری"],
    "cpt_delay":      ["still waiting", "not arrived", "abhi tak", "intezar", "late", "delayed", "delay",
                       "no delivery", "ابھی تک", "نہیں آیا", "نہیں ہوئی"],
    "cpt_available":  ["available", "stock", "store", "branch", "milega", "دستیاب", "اسٹاک"],
    "cpt_size":       ["size", "stitched", "unstitched", "chart", "medium", "سائز", "سلا"],
    "cpt_fabric":     ["fabric", "quality", "kapra", "fade", "rang", "wash", "dhone", "کپڑا", "کوالٹی"],
    "cpt_praise":     ["beautiful", "love", "stunning", "elegant", "khoobsurat", "pasand", "zabardast",
                       "pyara", "masha", "design", "خوبصورت", "پیارا", "ماشاء"],
    "cpt_intent":     ["to order", "order kar", "order karna", "order kaise", "buy", "inbox", "dm",
                       "lena hai", "place an order", "آرڈر", "ان باکس"],
}

def add_concepts(text):
    found = [name for name, words in CONCEPTS.items() if any(w in text for w in words)]
    return text + " " + " ".join(found * CONCEPT_WEIGHT)   # repeated so the concept counts more than one ordinary word

def make_vectors(texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    return vec, vec.fit_transform(texts)

# =========================================================================
print("=" * 72); print("STEP 5: CLUSTERING THE COMMENTS (K-Means)"); print("=" * 72)

# ---- 1. Experiment: text only vs text + concept tokens ---------------------
_, X_plain = make_vectors(comments["text_clean"])
vec, X = make_vectors(comments["text_clean"].apply(add_concepts))
km_plain = KMeans(K, n_init=5, random_state=SEED).fit(X_plain)
print(f"\n1) Text only         : ARI = {adjusted_rand_score(comments['topic'], km_plain.labels_):.2f}, "
      f"NMI = {normalized_mutual_info_score(comments['topic'], km_plain.labels_):.2f}")

# ---- 2. Choosing k -----------------------------------------------------------
print("\n2) Choosing k (silhouette is measured on a sample of 5,000 comments):")
sample = np.random.RandomState(0).choice(len(comments), 5000, replace=False)
print(f"   {'k':>3} {'inertia':>9} {'silhouette':>11}")
for k in range(6, 15):
    km_k = KMeans(k, n_init=5, random_state=SEED).fit(X)
    sil = silhouette_score(X[sample], km_k.labels_[sample])
    print(f"   {k:>3} {km_k.inertia_:>9.0f} {sil:>11.3f}" + ("   <- chosen" if k == K else ""))
print("   Silhouette keeps rising slowly because many comments are word-for-word identical.")
print(f"   So I do not pick k from one number: k={K} gives clear, separate, readable segments.")

# ---- 3. Final model -------------------------------------------------------
km = KMeans(K, n_init=10, random_state=SEED).fit(X)
comments["cluster"] = km.labels_
ari = adjusted_rand_score(comments["topic"], km.labels_)
nmi = normalized_mutual_info_score(comments["topic"], km.labels_)
print(f"\n3) Text + concept tokens (final model, k={K}): ARI = {ari:.2f}, NMI = {nmi:.2f}")
print("   ARI/NMI compare my clusters with the 'topic' column. The model never saw that column.")
print("   (1.0 = identical grouping, 0 = random). With real data you would not have this column.")

# stability: K-Means starts from random points, so run it with 5 different seeds and compare
aris = [adjusted_rand_score(comments["topic"], KMeans(K, n_init=10, random_state=sd).fit(X).labels_) for sd in range(5)]
print(f"   Stability over 5 different random seeds: ARI between {min(aris):.2f} and {max(aris):.2f}")
print("   IMPORTANT: this data is simulated from short templates, and the keyword list is mine.")
print("   So a high score shows that the PIPELINE works. Real comments are messier and the score")
print("   would be lower. Do not claim this accuracy for real Pakistani brands.")

# ---- 4. Describe every segment -------------------------------------------
# Segment names: in real work you read the top words and name each cluster yourself.
# Here the name is suggested from the most common reference topic inside the cluster.
FRIENDLY = {
    "price_inquiry": "Price-inquiry shoppers", "price_complaint": "Price-sensitive critics",
    "discount_request": "Discount seekers", "delivery_query": "Delivery-info seekers",
    "delivery_complaint": "Delayed-order complainers", "availability_city": "Availability / city askers",
    "sizing": "Size and fit askers", "fabric_quality": "Fabric-quality askers",
    "praise": "Fans (praise)", "purchase_intent": "Ready-to-buy",
}
terms = np.array(vec.get_feature_names_out())
rows = []
for cid in range(K):
    part = comments[comments["cluster"] == cid]
    top_topic = part["topic"].value_counts()
    centroid = km.cluster_centers_[cid]
    top_idx = [i for i in centroid.argsort()[::-1] if not terms[i].startswith("cpt_")][:8]
    rows.append({
        "cluster": cid, "n_comments": len(part), "share": len(part) / len(comments),
        "top_terms": ", ".join(terms[top_idx]),
        "main_topic": top_topic.index[0], "purity": top_topic.iloc[0] / len(part),
        "pct_negative": (part["sentiment"] == "Negative").mean(),
        "pct_high_intent": (part["purchase_intent"] == "High").mean(),
        "pct_roman_urdu": (part["language_type"] == "Roman Urdu").mean(),
        "pct_english": (part["language_type"] == "English").mean(),
        "pct_urdu": (part["language_type"] == "Urdu").mean(),
        "examples": " | ".join(part["comment_text"].value_counts().index[:3]),
    })
summary = pd.DataFrame(rows)
summary["segment_name"] = summary["main_topic"].map(FRIENDLY)
dup = summary["segment_name"].duplicated(keep=False)           # two clusters with the same name get (1), (2)
summary.loc[dup, "segment_name"] += " (" + (summary[dup].groupby("segment_name").cumcount() + 1).astype(str) + ")"
summary = summary.sort_values("n_comments", ascending=False).reset_index(drop=True)
comments["segment_name"] = comments["cluster"].map(summary.set_index("cluster")["segment_name"])

print("\n4) THE AUDIENCE SEGMENTS (largest first)")
for r in summary.itertuples():
    print(f"\n   [{r.cluster}] {r.segment_name:<28} {r.n_comments:>6,} comments ({r.share:.1%})   purity {r.purity:.0%}")
    print(f"       top words : {r.top_terms}")
    print(f"       examples  : {r.examples}")
    print(f"       negative {r.pct_negative:.0%} | high purchase intent {r.pct_high_intent:.0%} | "
          f"Roman Urdu {r.pct_roman_urdu:.0%} / English {r.pct_english:.0%} / Urdu {r.pct_urdu:.0%}")

print("\n   Cluster vs reference topic (rows = my clusters, % of the cluster):")
ct = pd.crosstab(comments["segment_name"], comments["topic"], normalize="index").round(2)
print(ct.to_string())

# ---- 5. Segment shares per post and per platform x campaign (link to Project 2) --
pivot = (comments.groupby(["post_id", "segment_name"]).size().unstack(fill_value=0))
post_feat = pivot.div(pivot.sum(axis=1), axis=0).add_prefix("share_")
post_feat["n_comments_sampled"] = pivot.sum(axis=1)
post_feat = post_feat.reset_index()

merged = comments.merge(posts[["post_id", "platform", "campaign", "content_type"]], on="post_id")
by_pc = (merged.groupby(["platform", "campaign", "segment_name"]).size()
         .groupby(level=[0, 1]).transform(lambda s: s / s.sum()).rename("share").reset_index())

print("\n5) Biggest segment on each platform:")
by_platform = merged.groupby(["platform", "segment_name"]).size().groupby(level=0).transform(lambda s: s / s.sum()).unstack()
for p in by_platform.index:
    top3 = by_platform.loc[p].sort_values(ascending=False).head(3)
    print(f"   {p:<10}: " + " | ".join(f"{n} {v:.0%}" for n, v in top3.items()))

# ---- 6. Charts ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
s = summary.sort_values("n_comments")
ax.barh(s["segment_name"], s["share"] * 100, color="#3b6fd4")
ax.set_xlabel("% of all comments"); ax.set_title("Audience segments found by K-Means")
plt.tight_layout(); plt.savefig(OUT / "q2_segment_sizes.png", dpi=150); plt.close()

svd = TruncatedSVD(2, random_state=SEED)
pts = svd.fit_transform(X[sample])
jitter = np.random.RandomState(1).normal(0, 0.01, pts.shape)     # many comments are identical, so spread them a little
fig, ax = plt.subplots(figsize=(8, 5.5))
for name in summary["segment_name"]:
    m = (comments["segment_name"].values[sample] == name)
    ax.scatter(*(pts[m] + jitter[m]).T, s=6, alpha=0.5, label=name)
ax.set_title("Comments in 2D (each dot = one comment, colour = segment)")
ax.legend(fontsize=7, markerscale=2, loc="best"); plt.tight_layout()
plt.savefig(OUT / "q2_scatter.png", dpi=150); plt.close()

by_platform.loc[:, summary["segment_name"]].plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab10")
plt.ylabel("Share of comments"); plt.title("Audience segments by platform"); plt.xticks(rotation=0)
plt.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left"); plt.tight_layout()
plt.savefig(OUT / "q2_by_platform.png", dpi=150); plt.close()

# ---- 7. Save -----------------------------------------------------------------
comments[["post_id", "comment_text", "text_clean", "language_type", "topic", "sentiment",
          "purchase_intent", "cluster", "segment_name"]].to_csv(CLEAN / "comments_clustered.csv", index=False, encoding="utf-8-sig")
summary.to_csv(CLEAN / "cluster_summary.csv", index=False, encoding="utf-8-sig")
by_pc.to_csv(CLEAN / "segment_share_by_platform_campaign.csv", index=False)
post_feat.to_csv(CLEAN / "post_segment_features.csv", index=False)
print("\nSaved in data/clean/: comments_clustered.csv, cluster_summary.csv, "
      "segment_share_by_platform_campaign.csv, post_segment_features.csv")
print("Saved charts in outputs/: q2_segment_sizes.png, q2_scatter.png, q2_by_platform.png")
