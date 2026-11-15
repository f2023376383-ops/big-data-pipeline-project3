"""
Pipeline stage "Project 2": ad performance and budget recommendation.

This is the code of the Project 2 notebook (generate_ads, summarize, recommend_budget), copied without changing
the logic, so the pipeline can run it and produce the same numbers as the Project 2 documentation
(seed 42, 10,000 simulated ad records, Rs. 1,000,000 budget). The data is SIMULATED.

Writes: data/clean/p2_platform_summary.csv         (CTR, CPC, CPA, CVR, ROAS per platform)
        data/clean/p2_platform_campaign_roas.csv   (ROAS per platform x campaign)
        data/clean/budget_recommendation.csv       (the Rs. 1,000,000 split, same file as in Project 2)

Run from the project root folder:
    python src/p2_budget.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"

# ---- simulation settings (illustrative assumptions, NOT real market data) -------------------------
PLATFORMS = {
    #            avg impressions, CTR,   CPC (Rs), conv. rate, avg order value (Rs), share of rows
    "Facebook":  dict(impr=8000,  ctr=0.011, cpc=18, cvr=0.022, aov=4200, share=0.25),
    "Instagram": dict(impr=7000,  ctr=0.014, cpc=22, cvr=0.028, aov=4800, share=0.25),
    "Google":    dict(impr=4000,  ctr=0.032, cpc=35, cvr=0.040, aov=5200, share=0.20),
    "TikTok":    dict(impr=12000, ctr=0.009, cpc=12, cvr=0.014, aov=3800, share=0.18),
    "YouTube":   dict(impr=9000,  ctr=0.006, cpc=15, cvr=0.010, aov=4500, share=0.12),
}
CAMPAIGN_EFFECT = {  # (impressions multiplier, conversion-rate multiplier)
    "Winter Collection": (1.0, 1.00), "Eid Collection": (1.1, 1.25),
    "Summer Lawn Sale": (1.0, 1.00),  "Brand Awareness": (1.5, 0.60),
    "11.11 Sale": (1.2, 1.30),        "12.12 Sale": (1.2, 1.25),
}
DEVICES = {"Mobile": (0.72, 0.95), "Desktop": (0.22, 1.20), "Tablet": (0.06, 1.00)}   # (probability, cvr multiplier)
AGES    = {"18-24": (0.30, 0.90), "25-34": (0.38, 1.10), "35-44": (0.20, 1.05), "45+": (0.12, 0.95)}
CITIES  = {"Lahore": .24, "Karachi": .28, "Islamabad": .12, "Rawalpindi": .08, "Faisalabad": .10, "Multan": .09, "Peshawar": .09}
COLUMNS = ["date", "campaign", "platform", "impressions", "clicks", "spend", "conversions", "revenue",
           "age_group", "gender", "city", "device"]
SUM_COLS = ["impressions", "clicks", "spend", "conversions", "revenue"]


def generate_ads(n, seed=42):
    """Create n simulated ad records (vectorised, so 10 million rows is possible)."""
    rng = np.random.default_rng(seed)
    plat_names = np.array(list(PLATFORMS))
    p_idx = rng.choice(len(plat_names), size=n, p=[PLATFORMS[p]["share"] for p in plat_names])
    par = lambda key: np.array([PLATFORMS[p][key] for p in plat_names], dtype=float)[p_idx]

    # --- dates and campaigns (calendar based) ---
    day  = rng.integers(0, 365, size=n)
    date = np.datetime64("2025-01-01") + day.astype("timedelta64[D]")
    month = date.astype("datetime64[M]").astype(int) % 12 + 1
    dom   = (date - date.astype("datetime64[M]")).astype(int) + 1
    camp_names = np.array(list(CAMPAIGN_EFFECT))
    camp = np.select(
        [month <= 2, month <= 4, month <= 8, month <= 10, (month == 11) & (dom <= 15), month == 11, (month == 12) & (dom <= 15)],
        ["Winter Collection", "Eid Collection", "Summer Lawn Sale", "Brand Awareness", "11.11 Sale", "Winter Collection", "12.12 Sale"],
        default="Winter Collection")
    c_idx = np.searchsorted(np.sort(camp_names), camp)           # integer code for each campaign
    sorted_names = np.sort(camp_names)
    impr_mult = np.array([CAMPAIGN_EFFECT[c][0] for c in sorted_names])[c_idx]
    cvr_mult  = np.array([CAMPAIGN_EFFECT[c][1] for c in sorted_names])[c_idx]

    # --- audience ---
    dev_names, age_names, city_names = list(DEVICES), list(AGES), list(CITIES)
    d_idx = rng.choice(len(dev_names),  size=n, p=[DEVICES[d][0] for d in dev_names])
    a_idx = rng.choice(len(age_names),  size=n, p=[AGES[a][0] for a in age_names])
    y_idx = rng.choice(len(city_names), size=n, p=list(CITIES.values()))
    g_idx = rng.choice(2, size=n, p=[0.62, 0.38])
    cvr_mult = cvr_mult * np.array([DEVICES[d][1] for d in dev_names])[d_idx] * np.array([AGES[a][1] for a in age_names])[a_idx]

    # --- funnel: impressions -> clicks -> conversions -> revenue ---
    noise = lambda s: rng.lognormal(-s * s / 2, s, size=n)          # noise with average 1
    impressions = np.maximum((par("impr") * impr_mult * noise(0.6)).astype(np.int64), 50)
    ctr    = np.clip(par("ctr") * noise(0.20), 0.0005, 0.30)
    clicks = rng.binomial(impressions, ctr)
    spend  = np.round(clicks * par("cpc") * noise(0.20), 2)
    cvr    = np.clip(par("cvr") * cvr_mult * noise(0.20), 0.0, 0.60)
    conversions = rng.binomial(clicks, cvr)
    revenue = np.round(conversions * par("aov") * noise(0.25), 2)

    return pd.DataFrame({
        "date": date.astype("datetime64[ns]"),
        "campaign": pd.Categorical(camp),
        "platform": pd.Categorical.from_codes(p_idx, plat_names),
        "impressions": impressions, "clicks": clicks, "spend": spend,
        "conversions": conversions, "revenue": revenue,
        "age_group": pd.Categorical.from_codes(a_idx, age_names),
        "gender": pd.Categorical.from_codes(g_idx, ["Female", "Male"]),
        "city": pd.Categorical.from_codes(y_idx, city_names),
        "device": pd.Categorical.from_codes(d_idx, dev_names),
    })[COLUMNS]


def safe_div(a, b):
    """Division that returns NaN (not an error, not 0) when the denominator is 0."""
    return a / b.where(b != 0)


def summarize(data, by):
    """Group the data and calculate every metric from the group TOTALS."""
    g = data.groupby(by, observed=True)[SUM_COLS].sum()
    g["records"] = data.groupby(by, observed=True).size()
    g["CTR"]  = safe_div(g["clicks"], g["impressions"])
    g["CPC"]  = safe_div(g["spend"], g["clicks"])
    g["CPA"]  = safe_div(g["spend"], g["conversions"])
    g["CVR"]  = safe_div(g["conversions"], g["clicks"])
    g["ROAS"] = safe_div(g["revenue"], g["spend"])
    return g


def recommend_budget(summary, total_budget=1_000_000, alpha=1.0, min_share=0.05, max_share=0.45):
    """Share proportional to ROAS**alpha, then forced between min_share and max_share."""
    score = summary["ROAS"] ** alpha
    fixed = {}
    while True:
        free = [p for p in score.index if p not in fixed]
        if not free:
            break
        remaining = 1 - sum(fixed.values())
        tentative = score[free] / score[free].sum() * remaining
        over, under = tentative[tentative > max_share + 1e-12], tentative[tentative < min_share - 1e-12]
        if over.empty and under.empty:
            break
        for p in over.index:  fixed[p] = max_share
        for p in under.index: fixed[p] = min_share
    share = pd.Series({p: fixed.get(p, np.nan) for p in score.index})
    free = share.isna()
    if free.any():
        share[free] = score[free] / score[free].sum() * (1 - share[~free].sum())
    out = pd.DataFrame({"historical_ROAS": summary["ROAS"], "current_spend_share": summary["spend"] / summary["spend"].sum(),
                        "recommended_share": share})
    out = out.sort_values("recommended_share", ascending=False)
    out["budget_Rs"] = (out["recommended_share"] * total_budget).round(-3)
    out.iloc[0, out.columns.get_loc("budget_Rs")] += total_budget - out["budget_Rs"].sum()   # rounding leftover goes to the top platform
    out["expected_revenue_Rs"] = out["budget_Rs"] * out["historical_ROAS"]
    return out.sort_values("recommended_share", ascending=False)


def run(total_budget=1_000_000):
    df = generate_ads(10_000, seed=42)
    platform_summary = summarize(df, "platform").sort_values("ROAS", ascending=False)
    rec = recommend_budget(platform_summary, total_budget, alpha=1.0)
    combo = summarize(df, ["platform", "campaign"]).reset_index()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    platform_summary.reset_index().to_csv(CLEAN_DIR / "p2_platform_summary.csv", index=False)
    combo[["platform", "campaign", "records", "spend", "revenue", "CPA", "ROAS"]].to_csv(
        CLEAN_DIR / "p2_platform_campaign_roas.csv", index=False)
    rec.to_csv(CLEAN_DIR / "budget_recommendation.csv")          # same file name and format as in Project 2

    print("=" * 72); print("STAGE P2: AD PERFORMANCE AND BUDGET (simulated data)"); print("=" * 72)
    print(f"Ad records: {len(df):,} | overall spend Rs. {df['spend'].sum():,.0f} | revenue Rs. {df['revenue'].sum():,.0f} "
          f"| ROAS {df['revenue'].sum() / df['spend'].sum():.2f}")
    print("\nResults per platform:")
    print(platform_summary[["records", "spend", "revenue", "CTR", "CPC", "CPA", "CVR", "ROAS"]].round(3).to_string())
    print(f"\nRecommended split of Rs. {total_budget:,.0f}:")
    print(rec.round(3).to_string())
    print(f"Total allocated: Rs. {rec['budget_Rs'].sum():,.0f} | expected revenue: Rs. {rec['expected_revenue_Rs'].sum():,.0f}")
    print("\nSaved: data/clean/p2_platform_summary.csv, p2_platform_campaign_roas.csv, budget_recommendation.csv")
    return platform_summary, rec, combo


if __name__ == "__main__":
    run()
