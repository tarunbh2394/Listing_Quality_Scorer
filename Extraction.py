"""
STEP 2 — DATA EXTRACTION
========================
Normalizes the raw API JSON into a consistent product schema.
This decouples the rest of the pipeline from the API's format.
"""
from typing import Any


def extract_product(raw: dict) -> dict:
    """Standardize a raw Amazon API hit into our internal schema."""
    return {
        "asin"           : str(raw.get("asin", "N/A")),
        "title"          : str(raw.get("title", "") or ""),
        "price"          : float(raw.get("price", 0) or 0),
        "original_price" : float(raw.get("original_price", 0) or 0),
        "rating"         : float(raw.get("rating", 0) or 0),
        "reviews"        : int(raw.get("reviews", 0) or 0),
        "image_url"      : str(raw.get("image_url", "") or ""),
        "is_prime"       : bool(raw.get("is_prime", False)),
        "url"            : str(raw.get("url", "") or ""),
    }


def extract_all(ingested: dict) -> dict:
    """Extract target + all competitors into the standard schema."""
    return {
        "target"     : extract_product(ingested["target"]),
        "competitors": [extract_product(c) for c in ingested["competitors"]],
        "source"     : ingested["source"],
        "comp_query" : ingested["comp_query"],
        "kws"        : ingested["kws"],
    }