"""
Project 3 - Step 2: Simulated data generator
Pakistan Social Audience & Campaign Intelligence Engine

Creates two files (same idea as Table A and Table B in the project description):
    data/raw/posts.csv      one row = one social-media post
    data/raw/comments.csv   one row = one comment on a post

IMPORTANT: this data is SIMULATED (made with Python, seed 42), exactly like Project 2.
It is NOT real data from Sapphire, Khaadi, Gul Ahmed or any other brand.
It shows the METHOD, not facts about the real market.
Every row has source = "simulated".

Run from the project root folder:
    python src/generate_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_POSTS = 3000                      # project asks for roughly 2,000 - 5,000 posts
COLLECTION_DATE = pd.Timestamp("2026-01-15")   # pretend date when the data was collected
MAX_COMMENTS_PER_POST = 12          # we keep a sample of comments per post (not all)

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1. Settings for POSTS (Table A)
# ----------------------------------------------------------------------------
BRANDS = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E", "Brand F"]
PLATFORMS = ["Instagram", "TikTok", "Facebook"]
PLATFORM_P = [0.45, 0.30, 0.25]

# campaign names are the same as in Project 2, so the projects can be joined later
CAMPAIGNS = ["Winter Collection", "Eid Collection", "Summer Lawn Sale", "11.11 Sale", "12.12 Sale"]
CAMPAIGN_P = [0.20, 0.25, 0.30, 0.12, 0.13]
CAMPAIGN_WINDOWS = {            # date ranges in 2025 (same calendar idea as Project 2)
    "Winter Collection": [("2025-01-01", "2025-02-28"), ("2025-11-16", "2025-12-31")],
    "Eid Collection":    [("2025-03-01", "2025-04-30")],
    "Summer Lawn Sale":  [("2025-05-01", "2025-08-31")],
    "11.11 Sale":        [("2025-11-01", "2025-11-15")],
    "12.12 Sale":        [("2025-12-01", "2025-12-15")],
}
CONTENT_TYPES = ["Reel", "Product Photo", "Influencer Post", "Styling Video",
                 "Sale Announcement", "Eid Campaign Post"]
CONTENT_P = [0.25, 0.22, 0.15, 0.13, 0.15, 0.10]

# how strongly each choice changes views (these are MY assumptions, not real facts)
PLATFORM_VIEWS_LOG_MEAN = {"Instagram": 9.5, "TikTok": 10.2, "Facebook": 9.0}   # log-scale
PLATFORM_LIKE_RATE = {"Instagram": 0.05, "TikTok": 0.07, "Facebook": 0.025}
PLATFORM_SHARE_RATE = {"Instagram": 0.04, "TikTok": 0.06, "Facebook": 0.09}
CONTENT_EFFECT = {"Reel": 1.6, "Product Photo": 0.7, "Influencer Post": 1.3,
                  "Styling Video": 1.2, "Sale Announcement": 1.0, "Eid Campaign Post": 1.1}
CAMPAIGN_EFFECT = {"Winter Collection": 0.9, "Eid Collection": 1.3, "Summer Lawn Sale": 1.0,
                   "11.11 Sale": 1.2, "12.12 Sale": 1.2}
BRAND_SIZE = dict(zip(BRANDS, [1.8, 1.4, 1.2, 1.0, 0.8, 0.6]))   # bigger brand, more views

CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar"]

# ----------------------------------------------------------------------------
# 2. Settings for COMMENTS (Table B)
# ----------------------------------------------------------------------------
# For every topic: templates in English (en), Roman Urdu (rur) and Urdu script (ur).
# {city} is replaced by a city name.
TEMPLATES = {
    "price_inquiry": {
        "en":  ["Price please?", "What is the price of this suit?", "How much for this?", "Price?", "Kindly tell the price"],
        "rur": ["Price kya hai?", "Price bata dein please", "Iski qeemat kitni hai?", "Price kia ha is ka?", "Rate kya hai ismein?"],
        "ur":  ["قیمت کیا ہے؟", "اس کی قیمت بتائیں", "یہ کتنے کا ہے؟"],
    },
    "price_complaint": {
        "en":  ["Too expensive for lawn", "Price is very high", "Prices keep going up every season"],
        "rur": ["Price bohat zyada hai", "Itna mehanga kyun hai?", "Qeemat bohot ziada hai ab to", "Bohat mehngi lawn hai yaar"],
        "ur":  ["قیمت بہت زیادہ ہے", "بہت مہنگا ہے"],
    },
    "discount_request": {
        "en":  ["Any discount on this suit?", "Is there any sale coming?", "Will there be a discount code?", "Any offer on this?"],
        "rur": ["Koi discount milega?", "Sale kab lag rahi hai?", "Discount code hai koi?", "Is pe koi offer hai?", "Kuch kam ho sakta hai price?"],
        "ur":  ["ڈسکاؤنٹ ملے گا؟", "سیل کب لگے گی؟"],
    },
    "delivery_query": {
        "en":  ["How many days for delivery?", "Do you deliver to {city}?", "What are the delivery charges?"],
        "rur": ["Delivery kitne din mein hogi?", "{city} mein delivery hoti hai?", "Delivery charges kitne hain?", "Delivery kitnay din lagein gay?"],
        "ur":  ["ڈیلیوری کتنے دن میں ہوگی؟", "{city} میں ڈیلیوری ہوتی ہے؟"],
    },
    "delivery_complaint": {
        "en":  ["Ordered last week, still waiting", "My order has not arrived yet", "Still no delivery, very disappointed", "Order delayed again"],
        "rur": ["Pichle hafte order kiya tha abhi tak nahi aya", "Delivery abhi tak nahi hui", "Parcel kab milega? bohat intezar hai", "Order late ho gaya phir se"],
        "ur":  ["پچھلے ہفتے آرڈر کیا تھا ابھی تک نہیں آیا", "ڈیلیوری ابھی تک نہیں ہوئی"],
    },
    "availability_city": {
        "en":  ["Is this available in {city}?", "Do you have a store in {city}?", "Is this still in stock?"],
        "rur": ["Kya ye {city} mein available hai?", "{city} mein store hai aap ka?", "Ye stock mein hai abhi?", "{city} branch mein milega?"],
        "ur":  ["کیا یہ {city} میں دستیاب ہے؟", "یہ ابھی اسٹاک میں ہے؟"],
    },
    "sizing": {
        "en":  ["What sizes are available?", "Is this unstitched or stitched?", "Do you have a size chart?"],
        "rur": ["Kon se sizes available hain?", "Ye stitched hai ya unstitched?", "Size chart bhej dein please", "Medium size hai?"],
        "ur":  ["کون سے سائز دستیاب ہیں؟", "یہ سلا ہوا ہے یا ان سلا؟"],
    },
    "fabric_quality": {
        "en":  ["What is the fabric?", "Is the lawn quality good?", "Does the color fade after washing?", "Fabric is not like the picture"],
        "rur": ["Fabric kon sa hai?", "Quality kesi hai lawn ki?", "Dhone ke baad rang jata hai?", "Kapra picture jesa nahi tha"],
        "ur":  ["کپڑا کون سا ہے؟", "کوالٹی کیسی ہے؟"],
    },
    "praise": {
        "en":  ["Very beautiful dress!", "Love this design", "Stunning colors", "So elegant, masha'Allah"],
        "rur": ["Bohat khoobsurat dress!", "Mujhe ye design bohat pasand aya", "Kya baat hai, zabardast", "Masha'Allah bohat pyara"],
        "ur":  ["بہت خوبصورت ڈریس!", "ماشاءاللہ بہت پیارا"],
    },
    "purchase_intent": {
        "en":  ["I want to order this", "Please send details in inbox", "How can I place an order?", "Want to buy, DM please"],
        "rur": ["Inbox me details send kar dein", "Mujhe ye order karna hai", "Order kaise karun?", "Lena hai, DM kar dein"],
        "ur":  ["ان باکس میں تفصیلات بھیج دیں", "مجھے یہ آرڈر کرنا ہے"],
    },
}
EMOJIS = ["", "", "", " 😍", " 🔥", " 👌", " 🙏", " 😔", "!!", "..."]

# sentiment and purchase intent that belong to each topic
TOPIC_SENTIMENT = {
    "price_inquiry": "Neutral", "price_complaint": "Negative", "discount_request": "Neutral",
    "delivery_query": "Neutral", "delivery_complaint": "Negative", "availability_city": "Neutral",
    "sizing": "Neutral", "fabric_quality": "Neutral", "praise": "Positive", "purchase_intent": "Positive",
}
TOPIC_INTENT = {
    "price_inquiry": "Possible inquiry", "price_complaint": "None", "discount_request": "Possible inquiry",
    "delivery_query": "Possible inquiry", "delivery_complaint": "Existing customer",
    "availability_city": "Possible inquiry", "sizing": "Possible inquiry", "fabric_quality": "Possible inquiry",
    "praise": "None", "purchase_intent": "High",
}
TOPICS = list(TEMPLATES)
BASE_TOPIC_WEIGHT = {   # how common each topic is before any adjustment
    "price_inquiry": 14, "price_complaint": 6, "discount_request": 10, "delivery_query": 8,
    "delivery_complaint": 5, "availability_city": 9, "sizing": 8, "fabric_quality": 8,
    "praise": 24, "purchase_intent": 8,
}
LANG_NAMES = {"en": "English", "rur": "Roman Urdu", "ur": "Urdu"}
LANG_P = {"en": 0.35, "rur": 0.55, "ur": 0.10}


def topic_weights(campaign, content_type, platform, anomaly_kind):
    """Topic probabilities for one post. Sale campaigns get more price and discount comments, etc."""
    w = dict(BASE_TOPIC_WEIGHT)
    if campaign in ("11.11 Sale", "12.12 Sale"):
        w["discount_request"] *= 2.5; w["price_inquiry"] *= 1.5; w["delivery_query"] *= 1.4
    if campaign == "Eid Collection":
        w["availability_city"] *= 1.6; w["sizing"] *= 1.5; w["praise"] *= 1.3
    if content_type == "Sale Announcement":
        w["discount_request"] *= 2.0; w["price_inquiry"] *= 1.5; w["price_complaint"] *= 1.5
    if content_type in ("Influencer Post", "Styling Video"):
        w["praise"] *= 1.4; w["fabric_quality"] *= 1.3
    if content_type == "Product Photo":
        w["price_inquiry"] *= 1.3; w["sizing"] *= 1.3
    if platform == "TikTok":
        w["praise"] *= 1.3
    if platform == "Facebook":
        w["delivery_complaint"] *= 1.5; w["delivery_query"] *= 1.3
    if platform == "Instagram":
        w["availability_city"] *= 1.3
    # injected unusual posts (so the anomaly detector in Step 5 has something real to find)
    if anomaly_kind == "delivery_surge":
        w["delivery_complaint"] *= 12
    elif anomaly_kind == "price_backlash":
        w["price_complaint"] *= 12
    p = np.array([w[t] for t in TOPICS], dtype=float)
    return p / p.sum()


def random_date(campaign):
    start, end = CAMPAIGN_WINDOWS[campaign][rng.integers(len(CAMPAIGN_WINDOWS[campaign]))]
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    return pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(0, days + 1)))


# ----------------------------------------------------------------------------
# 3. Build Table A (posts)
# ----------------------------------------------------------------------------
def make_posts(n):
    rows = []
    for i in range(1, n + 1):
        brand = rng.choice(BRANDS)
        platform = rng.choice(PLATFORMS, p=PLATFORM_P)
        campaign = rng.choice(CAMPAIGNS, p=CAMPAIGN_P)
        content = rng.choice(CONTENT_TYPES, p=CONTENT_P)
        post_date = random_date(campaign)

        # about 6% of posts are made unusual on purpose
        r = rng.random()
        anomaly = "viral" if r < 0.02 else "delivery_surge" if r < 0.04 else "price_backlash" if r < 0.06 else ""

        views = rng.lognormal(PLATFORM_VIEWS_LOG_MEAN[platform], 0.6)
        views *= CONTENT_EFFECT[content] * CAMPAIGN_EFFECT[campaign] * BRAND_SIZE[brand]
        if anomaly == "viral":
            views *= 6
        likes = views * PLATFORM_LIKE_RATE[platform] * rng.lognormal(0, 0.35)
        comments = likes * rng.beta(2, 20) * (3 if anomaly in ("delivery_surge", "price_backlash") else 1)
        shares = likes * PLATFORM_SHARE_RATE[platform] * rng.lognormal(0, 0.5)

        rows.append({
            "post_id": f"P{i:05d}", "brand": brand, "platform": platform, "campaign": campaign,
            "content_type": content, "product_category": "Lawn suit", "post_date": post_date.date(),
            "likes": int(likes), "comments_count": max(int(comments), 1), "shares": int(shares),
            "views": int(views),
            "days_since_post": (COLLECTION_DATE - post_date).days,
            "source": "simulated",
            "injected_anomaly": anomaly,   # ONLY for checking the detector later, never use as a feature
        })
    posts = pd.DataFrame(rows)
    # views are "if available": hide them for about 25% of Facebook posts
    hide = (posts["platform"] == "Facebook") & (rng.random(len(posts)) < 0.25)
    posts["views"] = posts["views"].astype("Int64")
    posts.loc[hide, "views"] = pd.NA
    return posts


# ----------------------------------------------------------------------------
# 4. Build Table B (comments)
# ----------------------------------------------------------------------------
def make_comment_text(topic, lang_key):
    text = str(rng.choice(TEMPLATES[topic][lang_key])).replace("{city}", str(rng.choice(CITIES)))
    if lang_key != "ur":                       # emojis and small style changes only for Latin script
        text += str(rng.choice(EMOJIS))
        if rng.random() < 0.2:
            text = text.lower()
    return text


def make_comments(posts):
    rows = []
    for p in posts.itertuples(index=False):
        n = min(p.comments_count, MAX_COMMENTS_PER_POST * (2 if p.injected_anomaly in
                ("delivery_surge", "price_backlash") else 1))
        probs = topic_weights(p.campaign, p.content_type, p.platform, p.injected_anomaly)
        for topic in rng.choice(TOPICS, size=n, p=probs):
            lang_key = rng.choice(list(LANG_P), p=list(LANG_P.values()))
            sentiment = TOPIC_SENTIMENT[topic]
            if topic == "fabric_quality":      # fabric comments can be good, bad or neutral
                sentiment = "Negative" if rng.random() < 0.3 else "Neutral"
            rows.append({
                "post_id": p.post_id, "comment_text": make_comment_text(topic, lang_key),
                "language_type": LANG_NAMES[lang_key], "topic": topic,
                "sentiment": sentiment, "purchase_intent": TOPIC_INTENT[topic],
                "source": "simulated",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)

    posts = make_posts(N_POSTS)
    comments = make_comments(posts)

    posts.to_csv(out / "posts.csv", index=False)
    comments.to_csv(out / "comments.csv", index=False, encoding="utf-8-sig")

    print(f"posts.csv    : {posts.shape[0]:,} rows, {posts.shape[1]} columns")
    print(f"comments.csv : {comments.shape[0]:,} rows, {comments.shape[1]} columns")
    print("\nPosts per platform:\n", posts["platform"].value_counts().to_string())
    print("\nComments per language:\n", comments["language_type"].value_counts().to_string())
    print("\nComments per topic:\n", comments["topic"].value_counts().to_string())
    print("\nMedian likes by content type:\n", posts.groupby("content_type")["likes"].median().round(0).to_string())
    print("\nSaved in:", out)
