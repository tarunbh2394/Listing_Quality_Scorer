"""
STEP 5 — MODEL PIPELINE
=======================
Runs LQS prediction + CTR/CVR/RPI estimation on every product.
Output: pandas DataFrame with all metrics, indexed by TARGET / COMP-N.
"""
import logging
import math

import joblib
import numpy as np
import pandas as pd

from config import CFG

log = logging.getLogger("opsell.pipeline")


def _load_model():
    if not CFG.model_path.exists():
        raise FileNotFoundError(f"Model not found: {CFG.model_path}")
    mdl  = joblib.load(CFG.model_path)
    cols = list(mdl.feature_names_in_)
    log.info("Model loaded | %d features", len(cols))
    return mdl, cols


MODEL, EXPECTED_COLS = _load_model()


def _heuristic_lqs(f: dict) -> float:
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


def predict_lqs(features: dict) -> tuple[float, str]:
    X = pd.DataFrame([{c: features.get(c, 0) for c in EXPECTED_COLS}])[EXPECTED_COLS]
    model_score = float(np.clip(MODEL.predict(X)[0], CFG.lqs_min, CFG.lqs_max))
    h_score     = CFG.lqs_min + (_heuristic_lqs(features) / 100) * (CFG.lqs_max - CFG.lqs_min)
    final       = CFG.blend_model_w * model_score + (1 - CFG.blend_model_w) * h_score
    final       = float(np.clip(final, CFG.lqs_min, CFG.lqs_max))
    grade       = ("A" if final >= CFG.grade_a_cut
                   else "B" if final >= CFG.grade_b_cut
                   else "C")
    return round(final, 1), grade


def estimate_ctr(p: dict, f: dict) -> float:
    s  = min(f["star_rating"] / 5.0, 1.0) * 30
    s += min(math.log1p(f["review_count"]) / math.log1p(5000), 1.0) * 25
    s += 10 if f["discount_pct"] > 10 else 0
    s += 10 if f["image_count"] >= 1  else 0
    s += 10 if p.get("is_prime")      else 0
    s += 15 if 0 < f["best_seller_rank"] <= 100 else 0
    return round(min(s / 100.0, 1.0), 3)


def estimate_cvr(p: dict, f: dict) -> float:
    s  = min(f["star_rating"] / 5.0, 1.0) * 35
    s += min(math.log1p(f["review_count"]) / math.log1p(5000), 1.0) * 30
    s += 15 if p.get("is_prime")           else 0
    s += 10 if f["discount_pct"] > 10      else 0
    s += 10 if f["has_aplus"] == 1         else 0
    s += 5  if f["has_video"]  == 1        else 0
    return round(min(s / 100.0, 1.0), 3)


def score_product(p: dict) -> dict:
    f = p["features"]
    lqs, grade = predict_lqs(f)
    ctr = estimate_ctr(p, f)
    cvr = estimate_cvr(p, f)
    asp = f["price"] if f["price"] > 0 else 1.0
    return {
        "title"   : p["title"][:65],
        "asin"    : p["asin"],
        "price"   : asp,
        "discount": f["discount_pct"],
        "rating"  : f["star_rating"],
        "reviews" : f["review_count"],
        "lqs"     : lqs,
        "grade"   : grade,
        "ctr"     : ctr,
        "cvr"     : cvr,
        "rpi"     : round(asp * ctr * cvr, 2),
        "qc_flags": p.get("qc_flags", []),
    }


def run_pipeline(feature_data: dict) -> pd.DataFrame:
    rows = [score_product(feature_data["target"])] + \
           [score_product(c) for c in feature_data["competitors"]]
    df = pd.DataFrame(rows)
    df.index = ["TARGET"] + [f"COMP-{i+1}" for i in range(len(feature_data["competitors"]))]
    return df