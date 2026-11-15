"""
Pipeline stage "Final": Question 5 - "What should the marketing team do next?"
This is where Project 1, Project 2 and Project 3 are joined.

For every platform x campaign it combines:
    Project 3 : which audience needs are biggest in the comments (segments), which posts are unusual,
                which content type works best on that platform
    Project 2 : ROAS and recommended budget share of that platform, ROAS of that platform in that campaign
    Project 1 : organic benchmark (how the platform ranks for unpaid posts in the public dataset)
and turns it into a short list of actions, each with evidence and a metric to test it.

The action rules are written by me (marketing common sense). The numbers come from the data.
All results come from SIMULATED data: they show the method, not facts about real brands.

Reads : files in data/clean/ made by the earlier stages
Writes: data/clean/final_recommendations.csv     (one row per platform x campaign x action)
        data/clean/platform_context.csv          (one row per platform: P1 + P2 + P3 facts together)
        outputs/marketing_recommendations.md     (readable report)

Run from the project root folder:
    python src/build_recommendations.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# What to do for each audience need. Key = main topic of the segment (from Step 5).
RULES = {
    "price_inquiry":      ("Content",          "Put the price in the first line of the caption and in a pinned comment, so people do not need to ask.",
                           "number of 'price?' comments per 100 comments"),
    "price_complaint":    ("Promotion",        "Test a value message (bundle, 2-piece offer, fabric proof) instead of a plain price cut.",
                           "share of negative price comments; sales of the bundle"),
    "discount_request":   ("Promotion",        "Run a time-limited discount code and announce it in the post, so the question is answered before it is asked.",
                           "code uses; discount-request comments per 100 comments"),
    "delivery_query":     ("Content",          "Add delivery time and charges to the caption or highlights, and set up an automatic reply to the question.",
                           "delivery-question comments per 100 comments"),
    "delivery_complaint": ("Customer service", "Give customer service a named person for this campaign: reply under the affected posts within 24 hours and check courier delays.",
                           "delayed-order complaints per 100 comments; first reply time"),
    "availability_city":  ("Content",          "Show city-wise availability or a store link in the post, and say clearly when stock is low.",
                           "'available in my city?' comments per 100 comments"),
    "sizing":             ("Content",          "Add a size chart and a stitched / unstitched explanation (carousel or story highlight).",
                           "size-question comments per 100 comments"),
    "fabric_quality":     ("Content",          "Post fabric close-ups and a wash or care test, and pin the fabric details.",
                           "fabric-question comments per 100 comments; share of negative fabric comments"),
    "praise":             ("Content",          "Reuse the most praised designs as social proof and ask fans to share their own photos.",
                           "shares per post; user photos received"),
    "purchase_intent":    ("Sales",            "Reply to 'inbox me' comments fast with a direct order link; track how many DMs become orders.",
                           "DM reply time; DM-to-order rate"),
}
NEGATIVE_TOPICS = {"price_complaint", "delivery_complaint"}
# How urgent is a segment? Score = share of comments x weight. Angry customers and ready buyers come first.
WEIGHT = lambda topic: 2.0 if topic in NEGATIVE_TOPICS else 1.5 if topic == "purchase_intent" else 0.5 if topic == "praise" else 1.0
ACTIONS_PER_COMBO = 3

NAME_FIX = {"Tiktok": "TikTok", "Linkedin": "LinkedIn"}      # Project 1 wrote "Tiktok" (title case)

# ----------------------------------------------------------------------------- load
seg_share = pd.read_csv(CLEAN / "segment_share_by_platform_campaign.csv")
clusters = pd.read_csv(CLEAN / "cluster_summary.csv", keep_default_na=False, na_values=[""])
flagged = pd.read_csv(CLEAN / "flagged_posts.csv")
content = pd.read_csv(CLEAN / "content_performance.csv")
budget = pd.read_csv(CLEAN / "budget_recommendation.csv").rename(columns={"Unnamed: 0": "platform"})
roas_pc = pd.read_csv(CLEAN / "p2_platform_campaign_roas.csv")
p1 = pd.read_csv(CLEAN / "p1_platform_benchmark.csv")
p1["platform"] = p1["platform"].replace(NAME_FIX)

topic_of = clusters.set_index("segment_name")["main_topic"]
seg_share["topic"] = seg_share["segment_name"].map(topic_of)

# ------------------------------------------------------------ platform facts (P1 + P2 + P3)
p1["p1_rank"] = p1["median_likes"].rank(ascending=False).astype(int)
best_content = (content.sort_values("median_index", ascending=False).groupby("platform").first()["content_type"]
                .rename("best_content_type"))
n_flag = flagged[flagged["is_unusual"]].groupby("platform").size().rename("unusual_posts")
n_posts = flagged.groupby("platform").size().rename("posts")
plat = (budget.set_index("platform")[["historical_ROAS", "recommended_share", "budget_Rs"]]
        .join(p1.set_index("platform")[["median_likes", "p1_rank"]].rename(columns={"median_likes": "p1_organic_median_likes"}))
        .join(best_content).join(n_flag).join(n_posts))
plat = plat.loc[plat.index.isin(seg_share.merge(flagged[["platform"]].drop_duplicates(), on="platform")["platform"])]
plat["unusual_share"] = plat["unusual_posts"] / plat["posts"]
plat.to_csv(CLEAN / "platform_context.csv")

# ------------------------------------------------------------------ recommendations
roas_lookup = roas_pc.set_index(["platform", "campaign"])
rows = []
for (platform, campaign), part in seg_share.groupby(["platform", "campaign"]):
    part = part.copy()
    part["score"] = part["share"] * part["topic"].map(WEIGHT)
    top = part.sort_values("score", ascending=False).head(ACTIONS_PER_COMBO)

    # context from Project 2
    p2_note = ""
    if (platform, campaign) in roas_lookup.index and platform in plat.index:
        r_c, r_p = roas_lookup.loc[(platform, campaign), "ROAS"], plat.loc[platform, "historical_ROAS"]
        verdict = ("strong campaign for this platform" if r_c >= 1.1 * r_p
                   else "weaker than usual for this platform" if r_c <= 0.9 * r_p else "about normal for this platform")
        p2_note = f"P2: ROAS {r_c:.1f} in this campaign vs {r_p:.1f} on {platform} overall ({verdict}); budget share {plat.loc[platform, 'recommended_share']:.0%}"
    unusual_here = flagged[(flagged["platform"] == platform) & (flagged["campaign"] == campaign) & flagged["is_unusual"]]

    for rank, r in enumerate(top.itertuples(), start=1):
        kind, action, metric = RULES[r.topic]
        rows.append({
            "platform": platform, "campaign": campaign, "priority": rank, "segment": r.segment_name,
            "segment_share": round(r.share, 3), "action_type": kind, "action": action, "metric_to_test": metric,
            "evidence": f"{r.share:.0%} of comments on {platform} / {campaign} posts are '{r.segment_name}'",
            "p2_context": p2_note,
            "best_content_type": plat.loc[platform, "best_content_type"] if platform in plat.index else "",
            "unusual_posts_here": len(unusual_here),
        })
rec = pd.DataFrame(rows)
rec.to_csv(CLEAN / "final_recommendations.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------------ report
lines = ["# Marketing recommendations (simulated data)", "",
         "**Read this first:** all numbers come from simulated data. They show how the method works, not facts about real brands.", "",
         "Each action has evidence (from the comments), a P2 budget note, and a metric to check after the test.", ""]
for platform in plat.index:
    pr = plat.loc[platform]
    lines += [f"## {platform}", "",
              f"- **Project 2 (paid ads):** ROAS {pr['historical_ROAS']:.2f}, recommended budget Rs. {pr['budget_Rs']:,.0f} ({pr['recommended_share']:.0%} of Rs. 1,000,000)",
              f"- **Project 1 (organic, public dataset):** median {pr['p1_organic_median_likes']:.0f} likes per post, rank {int(pr['p1_rank'])} of 5 platforms",
              f"- **Project 3 (this project):** best content type here is **{pr['best_content_type']}**; "
              f"{int(pr['unusual_posts'])} of {int(pr['posts'])} posts ({pr['unusual_share']:.1%}) flagged as unusual", ""]
    for campaign, part in rec[rec["platform"] == platform].groupby("campaign"):
        lines += [f"### {platform} / {campaign}", "", f"_{part['p2_context'].iloc[0]}_", ""]
        for r in part.sort_values("priority").itertuples():
            lines += [f"{r.priority}. **{r.action_type}: {r.segment}** ({r.segment_share:.0%} of comments)",
                      f"   - Action: {r.action}", f"   - Check with: {r.metric_to_test}", ""]
lines += ["## Limits", "",
          "- Data is simulated. Segments, ROAS and flags show the method, not the market.",
          "- ROAS is not profit, and expected revenue assumes ROAS stays the same when spend changes.",
          "- Comments show what people say, not why. Every action is a TEST to run for a few weeks, not a proven fix.",
          "- Project 1 data is a public dataset (not Pakistani brands) and Project 2 / 3 data is simulated, so the three projects are joined by method and by platform / campaign names, not by the same real customers."]
(OUT / "marketing_recommendations.md").write_text("\n".join(lines), encoding="utf-8")

# ----------------------------------------------------------------------- print
print("=" * 72); print("FINAL STAGE: RECOMMENDATIONS (P1 + P2 + P3 joined)"); print("=" * 72)
print("\nPlatform facts from all three projects:")
print(plat[["historical_ROAS", "recommended_share", "p1_organic_median_likes", "p1_rank", "best_content_type",
            "unusual_posts", "unusual_share"]].round(3).to_string())
print("\nMost common actions over all platform x campaign combinations:")
print(rec.groupby(["action_type", "segment"]).size().sort_values(ascending=False).head(8).to_string())
sample = rec[(rec["platform"] == "TikTok") & (rec["campaign"] == "11.11 Sale")].sort_values("priority")
print("\nExample: TikTok / 11.11 Sale")
for r in sample.itertuples():
    print(f"  {r.priority}. [{r.action_type}] {r.segment} ({r.segment_share:.0%})\n     {r.action}")
print(f"  {sample['p2_context'].iloc[0]}")
print(f"\nSaved: data/clean/final_recommendations.csv ({len(rec)} rows), data/clean/platform_context.csv")
print("Saved: outputs/marketing_recommendations.md")
