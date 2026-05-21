"""
Self-contained Ohria comparison — uses omkarcloud API + LQS model directly.
No dependency on Ingestion.py / Extraction.py / etc.
"""
import math
import re

import joblib
import numpy as np
import pandas as pd
import requests

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — change only if your paths differ
# ════════════════════════════════════════════════════════════════════════════
API_KEY      = "ok_7a7f092c1855099ab70464867575fc29"
BASE_URL     = "https://amazon-scraper-api.omkar.cloud"
COUNTRY_CODE = "IN"
HEADERS      = {"API-Key": API_KEY}
MODEL_PATH   = r"C:\Users\tarun\OneDrive\Desktop\Opsell\LQS_Project\models\lqs_model.pkl"

TARGETS = [
    {"name": "Ohria Coconut Cleanser Shampoo", "asin": "B08RCS9PXP",
     "comp_query": "ayurvedic shampoo coconut"},
    {"name": "Ohria Pomegranate Face Cream",   "asin": "B08TDXHS9Q",
     "comp_query": "ayurvedic face cream"},
]

# ════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════════════════════════
print("Loading model...")
MODEL = joblib.load(MODEL_PATH)
EXPECTED_COLS = list(MODEL.feature_names_in_)
print(f"✓ Model loaded | {len(EXPECTED_COLS)} features\n")

BRAND_VOCAB = {
    "hp","dell","lenovo","asus","acer","apple","samsung","msi","sony","bose",
    "boat","jbl","noise","mamaearth","wow","plum","biotique","khadi","forest",
    "ohria","ayurveda","ayurvedic","himalaya","dabur","patanjali","kama",
    "lakshya","just","herbals","forest essentials","oriflame",
}


# ════════════════════════════════════════════════════════════════════════════
# FETCH
# ════════════════════════════════════════════════════════════════════════════
def fetch_page(query, page=1):
    try:
        r = requests.get(f"{BASE_URL}/amazon/search",
                         params={"query": query, "country_code": COUNTRY_CODE, "page": page},
                         headers=HEADERS, timeout=60)
        return r.json().get("results", [])
    except Exception as e:
        print(f"  page {page} error: {e}")
        return []


def fetch_by_asin(asin):
    results = fetch_page(asin, page=1)
    for r in results:
        if str(r.get("asin", "")).upper() == asin.upper():
            return r
    if results:
        print(f"  ⚠ Exact ASIN {asin} not found, using closest match.")
        return results[0]
    raise RuntimeError(f"ASIN {asin} returned no results.")


def fetch_competitors(query, exclude_asin, n=5):
    results = fetch_page(query, page=1)
    return [r for r in results if r.get("asin") != exclude_asin][:n]


# ════════════════════════════════════════════════════════════════════════════
# FEATURES + LQS
# ════════════════════════════════════════════════════════════════════════════
def extract_features(p):
    title  = str(p.get("title", "") or "")
    price  = float(p.get("price", 0) or 0)
    mrp    = float(p.get("original_price", 0) or 0)
    rating = float(p.get("rating", 0) or 0)
    rev    = float(p.get("reviews", 0) or 0)
    disc   = round((mrp - price) / mrp * 100, 2) if mrp > 0 else 0

    tl = title.lower()
    brand_present  = 1 if any(b in tl for b in BRAND_VOCAB) else 0
    sentence_case  = 1 if title and title[0].isupper() else 0
    is_established = (rating >= 4.0 and rev >= 100)

    bullet_count = 5   if is_established else 2
    desc_words   = 200 if is_established else 40
    image_count  = 7   if is_established else 3
    has_aplus    = 1   if (brand_present and rev >= 100) else 0
    has_video    = 1   if (brand_present and rev >= 500) else 0
    bsr_proxy    = max(1, int(50000 / (rev + 10)))

    return {
        "title_length": len(title), "title_word_count": len(title.split()),
        "title_brand_present": brand_present, "title_sentence_case": sentence_case,
        "title_spellcheck_pass": 1, "bullet_count": bullet_count,
        "bullets_spell_pass": 1, "bullets_caps_pass": 1,
        "final_bullet_flag": 1 if bullet_count >= 4 else 0,
        "description_word_count": desc_words,
        "final_desc_flag": 1 if desc_words >= 100 else 0,
        "price": price, "mrp": mrp, "discount_pct": disc,
        "star_rating": rating, "review_count": rev,
        "log_review_count": np.log1p(rev),
        "best_seller_rank": bsr_proxy, "image_count": image_count,
        "video_count": 1 if has_video else 0, "has_video": has_video,
        "has_aplus": has_aplus, "aplus_check_pass": has_aplus,
    }


def heuristic_lqs(f):
    s = 0.0
    s += min(f["star_rating"] / 5.0, 1.0) * 30
    s += min(np.log1p(f["review_count"]) / np.log1p(5000), 1.0) * 30
    s += 10 if f["title_brand_present"] else 0
    s += 10 if f["discount_pct"] > 10 else (5 if f["discount_pct"] > 0 else 0)
    if 80 <= f["title_length"] <= 200: s += 10
    elif f["title_length"] >= 50:      s += 5
    s += 5 if f["image_count"] >= 1 else 0
    s += 5 if (f["star_rating"] >= 4.0 and f["review_count"] >= 100) else 0
    return s


def predict_lqs(p):
    f = extract_features(p)
    X = pd.DataFrame([{c: f.get(c, 0) for c in EXPECTED_COLS}])[EXPECTED_COLS]
    model_score = float(np.clip(MODEL.predict(X)[0], 65, 95))
    h_score = 65 + (heuristic_lqs(f) / 100) * 30
    final = float(np.clip(0.4 * model_score + 0.6 * h_score, 65, 95))
    grade = "A" if final >= 88 else ("B" if final >= 75 else "C")
    return round(final, 1), grade, f


# ════════════════════════════════════════════════════════════════════════════
# RPI
# ════════════════════════════════════════════════════════════════════════════
def estimate_ctr(p, f):
    s = (min(f["star_rating"]/5.0, 1.0) * 30
         + min(math.log1p(f["review_count"])/math.log1p(5000), 1.0) * 25
         + (10 if f["discount_pct"] > 10 else 0)
         + (10 if f["image_count"] >= 1 else 0)
         + (10 if p.get("is_prime") else 0))
    return round(min(s/100, 1.0), 3)


def estimate_cvr(p, f):
    s = (min(f["star_rating"]/5.0, 1.0) * 35
         + min(math.log1p(f["review_count"])/math.log1p(5000), 1.0) * 30
         + (15 if p.get("is_prime") else 0)
         + (10 if f["discount_pct"] > 10 else 0))
    return round(min(s/100, 1.0), 3)


def score_product(p):
    lqs, grade, f = predict_lqs(p)
    asp = f["price"] if f["price"] > 0 else 1.0
    ctr = estimate_ctr(p, f)
    cvr = estimate_cvr(p, f)
    return {
        "title"   : str(p.get("title", ""))[:60],
        "asin"    : p.get("asin", "N/A"),
        "price"   : asp,
        "discount": f["discount_pct"],
        "rating"  : f["star_rating"],
        "reviews" : f["review_count"],
        "lqs"     : lqs,
        "grade"   : grade,
        "ctr"     : ctr,
        "cvr"     : cvr,
        "rpi"     : round(asp * ctr * cvr, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
# ANALYSE ONE TARGET
# ════════════════════════════════════════════════════════════════════════════
def analyse(target):
    print("\n" + "█" * 78)
    print(f"  ★ {target['name']}  (ASIN: {target['asin']})")
    print("█" * 78)

    print(f"  Fetching target...")
    target_prod = fetch_by_asin(target["asin"])
    print(f"  ✓ {str(target_prod.get('title',''))[:70]}")

    print(f"  Fetching top 5 competitors for query: '{target['comp_query']}'...")
    comps = fetch_competitors(target["comp_query"], target["asin"], n=5)
    print(f"  ✓ {len(comps)} competitors found")

    rows = [score_product(target_prod)] + [score_product(c) for c in comps]
    df = pd.DataFrame(rows)
    df.index = ["TARGET"] + [f"COMP-{i+1}" for i in range(len(comps))]

    print("\n  SCORES:")
    print(df[["title","price","discount","rating","reviews",
              "lqs","grade","ctr","cvr","rpi"]].to_string())

    if len(df) > 1:
        t    = df.loc["TARGET"]
        best = df.drop("TARGET").loc[df.drop("TARGET")["rpi"].idxmax()]
        print(f"\n  GAP TO BEST COMPETITOR ({best.name}):")
        for m in ["lqs", "ctr", "cvr", "rpi", "rating", "reviews"]:
            d = best[m] - t[m]
            tag = "worse" if d > 0 else ("better" if d < 0 else "same")
            print(f"    {m.upper():<8}  Target={t[m]:>10.3g}   Best={best[m]:>10.3g}   Gap={d:>+10.3g}  ({tag})")

    return df


# ════════════════════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════════════════════
results = {}
for tgt in TARGETS:
    try:
        results[tgt["name"]] = analyse(tgt)
    except Exception as e:
        import traceback
        print(f"\n  ⚠ ERROR analysing {tgt['name']}:")
        traceback.print_exc()

# ── Side-by-side summary ────────────────────────────────────────────────
print("\n" + "═" * 78)
print("  SIDE-BY-SIDE SUMMARY")
print("═" * 78)
print(f"  {'Product':<35} {'LQS':>6} {'Grade':>6} {'CTR':>6} {'CVR':>6} {'RPI':>10}")
print("  " + "─" * 70)
for name, df in results.items():
    t = df.loc["TARGET"]
    print(f"  {name[:34]:<35} {t['lqs']:>6} {t['grade']:>6} {t['ctr']:>6.3f} {t['cvr']:>6.3f} ₹{t['rpi']:>9.2f}")
    others = df.drop("TARGET")
    if not others.empty:
        best = others.loc[others["rpi"].idxmax()]
        print(f"  {'  ↳ best competitor':<35} {best['lqs']:>6} {best['grade']:>6} {best['ctr']:>6.3f} {best['cvr']:>6.3f} ₹{best['rpi']:>9.2f}")
        gap = best["rpi"] - t["rpi"]
        sign = "+" if gap >= 0 else ""
        print(f"  {'  → RPI gap':<35} {'':>26} ₹{sign}{gap:>9.2f}")
    print()
print("═" * 78)
print("\n✓ Analysis complete.")