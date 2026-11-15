"""
Project 3 - Step 9: Web app (Streamlit)
Pakistan Social Audience and Campaign Intelligence Engine

The app only SHOWS results. All calculations are done by the pipeline stages in src/.

Run from the project root folder:
    python run_pipeline.py        (once, to create the data)
    streamlit run app.py          (opens the website in your browser)
"""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CLEAN, OUT = ROOT / "data" / "clean", ROOT / "outputs"

st.set_page_config(page_title="Audience & Campaign Intelligence Engine", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    need = ["posts_scored.csv", "comments_clustered.csv", "cluster_summary.csv", "flagged_posts.csv",
            "final_recommendations.csv", "platform_context.csv", "budget_recommendation.csv",
            "p2_platform_campaign_roas.csv", "p1_platform_benchmark.csv"]
    missing = [f for f in need if not (CLEAN / f).exists()]
    if missing:
        return None, missing
    d = {
        "posts": pd.read_csv(CLEAN / "posts_scored.csv"),
        "comments": pd.read_csv(CLEAN / "comments_clustered.csv", keep_default_na=False, na_values=[""]),
        "clusters": pd.read_csv(CLEAN / "cluster_summary.csv", keep_default_na=False, na_values=[""]),
        "flagged": pd.read_csv(CLEAN / "flagged_posts.csv"),
        "rec": pd.read_csv(CLEAN / "final_recommendations.csv", keep_default_na=False, na_values=[""]),
        "plat": pd.read_csv(CLEAN / "platform_context.csv"),
        "budget": pd.read_csv(CLEAN / "budget_recommendation.csv").rename(columns={"Unnamed: 0": "platform"}),
        "roas": pd.read_csv(CLEAN / "p2_platform_campaign_roas.csv"),
        "p1": pd.read_csv(CLEAN / "p1_platform_benchmark.csv"),
    }
    d["comments"] = d["comments"].merge(d["posts"][["post_id", "platform", "campaign", "content_type"]], on="post_id")
    return d, []


data, missing = load()
if data is None:
    st.error("The result files are missing: " + ", ".join(missing) +
             ". Run `python run_pipeline.py` in the project folder first, then refresh this page.")
    st.stop()

posts, comments, clusters = data["posts"], data["comments"], data["clusters"]
flagged, rec = data["flagged"], data["rec"]

# ------------------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
all_platforms = sorted(posts["platform"].unique())
all_campaigns = sorted(posts["campaign"].unique())
sel_platforms = st.sidebar.multiselect("Platform", all_platforms, default=all_platforms)
sel_campaigns = st.sidebar.multiselect("Campaign", all_campaigns, default=all_campaigns)
st.sidebar.caption("The filters change the tabs for Q1 to Q5. The Pipeline tab always shows everything.")
st.sidebar.warning("All data is SIMULATED (seed 42). The app shows the method, not facts about real brands.")

p_f = posts[posts["platform"].isin(sel_platforms) & posts["campaign"].isin(sel_campaigns)]
c_f = comments[comments["platform"].isin(sel_platforms) & comments["campaign"].isin(sel_campaigns)]
f_f = flagged[flagged["platform"].isin(sel_platforms) & flagged["campaign"].isin(sel_campaigns)]
r_f = rec[rec["platform"].isin(sel_platforms) & rec["campaign"].isin(sel_campaigns)]

st.title("📊 Pakistan Social Audience & Campaign Intelligence Engine")
st.caption("Big Data Analytics, Project 3. Joins Project 1 (organic benchmark), Project 2 (ad budget) and Project 3 (audience) in one pipeline.")

tabs = st.tabs(["🔗 Pipeline", "1 · Content", "2 · Audience wants", "3 · Unusual posts",
                "4 · Audience needs", "5 · What to do", "⚠️ Limits"])

# ----------------------------------------------------------------------- Pipeline
with tabs[0]:
    st.subheader("How the three projects are connected")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
      P1 [label="Project 1\\norganic benchmark\\n(public dataset)"];
      P2 [label="Project 2\\nad performance and\\nbudget (simulated)"];
      D  [label="Project 3\\nsimulated posts\\nand comments"];
      C  [label="Cleaning and\\nquality checks"];
      Q1 [label="Q1 content\\nperformance"];
      Q2 [label="Q2/Q4 comment\\nclustering"];
      Q3 [label="Q3 unusual\\nposts"];
      R  [label="Q5 recommendations", fillcolor="#fdeccf"];
      W  [label="This website", fillcolor="#d9f2dd"];
      D -> C -> Q1; C -> Q2; C -> Q3; Q2 -> Q3;
      Q1 -> R; Q2 -> R; Q3 -> R; P1 -> R; P2 -> R; R -> W;
    }""")
    st.markdown("The projects are joined by **platform** and **campaign** names (Eid Collection, 11.11 Sale, 12.12 Sale ...), "
                "not by the same real customers: Project 1 is a public dataset, Projects 2 and 3 are simulated.")
    a, b, c, d = st.columns(4)
    a.metric("Posts", f"{len(posts):,}")
    b.metric("Comments", f"{len(comments):,}")
    c.metric("Audience segments", len(clusters))
    d.metric("Recommendations", len(rec))
    st.markdown("**Project 2 result used here: budget split of Rs. 1,000,000**")
    st.dataframe(data["budget"][["platform", "historical_ROAS", "recommended_share", "budget_Rs"]].round(3), hide_index=True)
    st.markdown("**Project 1 result used here: organic median likes per platform (public dataset)**")
    st.dataframe(data["p1"], hide_index=True)
    st.code("python run_pipeline.py   # runs all 8 stages\nstreamlit run app.py     # starts this website", language="bat")

# --------------------------------------------------------------------------- Q1
with tabs[1]:
    st.subheader("Q1. Which content attracts attention?")
    st.caption("Performance index = likes divided by the typical post of the same brand on the same platform. 1.0 is normal, 1.5 is 50% better.")
    if p_f.empty:
        st.info("No posts for this filter.")
    else:
        by_ct = (p_f.groupby("content_type").agg(posts=("post_id", "size"), median_likes=("likes", "median"),
                                                 median_index=("performance_index", "median"),
                                                 median_comments=("comments_count", "median"),
                                                 median_shares=("shares", "median"))
                 .sort_values("median_index", ascending=False).round(2))
        left, right = st.columns([3, 2])
        left.bar_chart(by_ct["median_index"], y_label="median performance index")
        right.dataframe(by_ct)
        best, worst = by_ct.index[0], by_ct.index[-1]
        st.success(f"In this selection **{best}** performs best (index {by_ct.loc[best, 'median_index']:.2f}) "
                   f"and **{worst}** weakest ({by_ct.loc[worst, 'median_index']:.2f}).")
        st.markdown("**Content type x platform** (median performance index)")
        heat = p_f.pivot_table(index="content_type", columns="platform", values="performance_index", aggfunc="median").round(2)
        st.dataframe(heat.style.background_gradient(cmap="YlGnBu", axis=None))
        st.caption("Careful: the engagement rate (likes + comments + shares per view) is almost the same for every content type here, "
                   "because in the simulation content type changes reach (views), not how much viewers like a post.")

# --------------------------------------------------------------------------- Q2
with tabs[2]:
    st.subheader("Q2. What does the audience want? (comment clusters)")
    st.caption("K-Means groups similar comments (English, Roman Urdu and Urdu together) into audience segments.")
    if c_f.empty:
        st.info("No comments for this filter.")
    else:
        seg = (c_f.groupby("segment_name").agg(comments=("post_id", "size"),
                                               negative=("sentiment", lambda s: (s == "Negative").mean()),
                                               high_intent=("purchase_intent", lambda s: (s == "High").mean()))
               .sort_values("comments", ascending=False))
        seg["share"] = seg["comments"] / seg["comments"].sum()
        left, right = st.columns([3, 2])
        left.bar_chart(seg["share"], y_label="share of comments")
        right.dataframe(seg.round(2))
        st.markdown("**Look inside a segment**")
        pick = st.selectbox("Segment", seg.index.tolist())
        info = clusters[clusters["segment_name"] == pick].iloc[0]
        st.write(f"Top words: `{info['top_terms']}`")
        sample_n = min(8, int((c_f["segment_name"] == pick).sum()))
        st.table(c_f[c_f["segment_name"] == pick].sample(sample_n, random_state=1)[["comment_text", "language_type", "platform", "sentiment"]])
        st.caption("Clusters are found without the topic column. Segment names are suggested from the most common reference topic inside each cluster.")

# --------------------------------------------------------------------------- Q3
with tabs[3]:
    st.subheader("Q3. Which marketing patterns are unusual?")
    st.caption("Each post is compared with posts of the same platform and content type. A flag means 'please look at this post'.")
    unusual = f_f[f_f["is_unusual"]].sort_values("strength", ascending=False)
    a, b, c = st.columns(3)
    a.metric("Posts in selection", f"{len(f_f):,}")
    b.metric("Flagged as unusual", f"{len(unusual):,}")
    c.metric("Share flagged", f"{len(unusual) / max(len(f_f), 1):.1%}")
    if unusual.empty:
        st.info("No unusual posts for this filter.")
    else:
        left, right = st.columns([2, 3])
        left.markdown("**Why they were flagged**")
        left.bar_chart(unusual["reason"].str.split(":").str[0].value_counts())
        right.markdown("**Reach score and anger score of every post**")
        plot = f_f.assign(status=f_f["is_unusual"].map({True: "flagged", False: "normal"}))
        right.scatter_chart(plot, x="z_reach", y="z_anger", color="status", x_label="reach score", y_label="anger score")
        st.markdown("**Posts to investigate first**")
        st.dataframe(unusual[["post_id", "brand", "platform", "campaign", "content_type", "likes", "comments_count",
                              "share_negative", "reason"]].head(25).round(2), hide_index=True)

# --------------------------------------------------------------------------- Q4
with tabs[4]:
    st.subheader("Q4. Which audience needs should the brand address?")
    st.caption("Share of comments per segment for each platform and campaign. Angry segments need a fix, curious ones need better information.")
    if c_f.empty:
        st.info("No comments for this filter.")
    else:
        only_neg = st.checkbox("Show only the angry segments (negative comments)")
        view = c_f[c_f["sentiment"] == "Negative"] if only_neg else c_f
        tab = (view.groupby(["platform", "campaign", "segment_name"]).size().groupby(level=[0, 1]).transform(lambda s: s / s.sum())
               .unstack(fill_value=0))
        st.dataframe(tab.style.format("{:.0%}").background_gradient(cmap="Oranges", axis=None))
        by_plat = view.groupby(["platform", "segment_name"]).size().groupby(level=0).transform(lambda s: s / s.sum()).unstack(fill_value=0)
        st.markdown("**Segments by platform**")
        st.bar_chart(by_plat)
        top = view["segment_name"].value_counts(normalize=True).head(3)
        st.info("Biggest needs in this selection: " + ", ".join(f"**{n}** ({v:.0%})" for n, v in top.items()))

# --------------------------------------------------------------------------- Q5
with tabs[5]:
    st.subheader("Q5. What should the marketing team do next?")
    st.caption("Actions combine the audience segments (P3), ROAS and budget (P2) and the organic benchmark (P1). Every action is a test, not a proven fix.")
    if r_f.empty:
        st.info("No recommendations for this filter.")
    else:
        types = st.multiselect("Action type", sorted(r_f["action_type"].unique()), default=sorted(r_f["action_type"].unique()))
        r_view = r_f[r_f["action_type"].isin(types)]
        for (plat, camp), part in r_view.groupby(["platform", "campaign"]):
            with st.expander(f"{plat}  /  {camp}"):
                st.markdown(f"_{part['p2_context'].iloc[0]}_  |  Best content type: **{part['best_content_type'].iloc[0]}**  |  "
                            f"Unusual posts here: {int(part['unusual_posts_here'].iloc[0])}")
                for r in part.sort_values("priority").itertuples():
                    st.markdown(f"**{r.priority}. {r.action_type}: {r.segment}** ({r.segment_share:.0%} of comments)")
                    st.write(r.action)
                    st.caption(f"Check with: {r.metric_to_test}")
        st.download_button("Download these recommendations (CSV)", r_view.to_csv(index=False).encode("utf-8-sig"),
                           file_name="recommendations.csv", mime="text/csv")
        md = OUT / "marketing_recommendations.md"
        if md.exists():
            st.download_button("Download the full report (Markdown)", md.read_bytes(), file_name="marketing_recommendations.md")

# ------------------------------------------------------------------------ Limits
with tabs[6]:
    st.subheader("Data and limits")
    st.markdown("""
- **Simulated data.** Posts and comments (Project 3) and ad records (Project 2) are generated with Python (seed 42). Every row has `source = simulated`.
  Real Instagram/TikTok comments were not downloaded because that is against their terms.
- **Project 1** uses a public dataset (not Pakistani brands). It is only an outside benchmark.
- **Keyword list.** The comment clustering uses a keyword list written by me (price, delivery, size ...). Without it, comments split by language, not by need.
  A high match with the reference topics shows that the pipeline works, not that real comments would be this easy.
- **Unusual posts** are flagged for a human to look at. Posts with few comments are hard to judge, so some unusual posts are missed.
- **ROAS is not profit.** Expected revenue assumes that ROAS stays the same when the budget changes, which is not guaranteed.
- **Comments show what people say, not why.** Clusters do not prove cause and effect. Treat every recommendation as a test.
- **Joining the projects** is done by platform and campaign names, not by the same customers.
""")
    quality = OUT / "data_quality_report.txt"
    if quality.exists():
        with st.expander("Data quality report"):
            st.text(quality.read_text(encoding="utf-8"))
    for img in ["q1_content_type.png", "q2_scatter.png", "q3_unusual_posts.png"]:
        if (OUT / img).exists():
            st.image(str(OUT / img), caption=img)
