"""
 Step 1- INGESTION
==================
Fetches raw Amazon listings via the omkarcloud API.
Inputs: ASIN | URL | search query.
Output: dict (target) + list of dicts (competitors).
"""
from __future__ import annotations

import logging
import random
import re
from typing import Optional

import requests

from config import CFG, CATEGORY_KEYWORDS

log = logging.getLogger("opsell.ingestion")

ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")


def asin_from_url(url: str) -> Optional[str]:
    m = ASIN_RE.search(url or "")
    return m.group(1) if m else None


def infer_category(text: str) -> tuple[Optional[str], list[str]]:
    q = (text or "").lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return cat, kws
    return None, []


def _matches(product: dict, kws: list[str]) -> bool:
    if not kws: return True
    return any(kw in str(product.get("title", "")).lower() for kw in kws)


class AmazonIngester:
    """Wraps the search API. All raw fetches go through here."""

    def _search(self, query: str, page: int = 1) -> list[dict]:
        try:
            r = requests.get(
                f"{CFG.base_url}/amazon/search",
                params={"query": query, "country_code": CFG.country_code, "page": page},
                headers=CFG.headers, timeout=CFG.timeout_s,
            )
            r.raise_for_status()
            return r.json().get("results", [])
        except requests.RequestException as e:
            log.warning("API error (page=%s): %s", page, e)
            return []

    def fetch_by_asin(self, asin: str) -> dict:
        asin = asin.strip().upper()
        results = self._search(asin, page=1)
        if not results:
            raise RuntimeError(f"ASIN '{asin}' returned no results.")
        for r in results:
            if str(r.get("asin", "")).upper() == asin:
                return r
        log.warning("Exact ASIN '%s' not found, using closest match.", asin)
        return results[0]

    def fetch_buried(self, query: str, kws: list[str], max_attempts: int = 8) -> tuple[dict, int]:
        pool = list(CFG.buried_pages)
        random.shuffle(pool)
        for i in range(max_attempts):
            page = random.choice(pool)
            results = self._search(query, page)
            valid   = [r for r in results if _matches(r, kws)]
            if valid:
                return random.choice(valid), page
            log.info("Attempt %d/%d: 0 matched on page %d", i+1, max_attempts, page)
        raise RuntimeError(f"No category-matched product for '{query}'.")

    def fetch_competitors(self, query: str, exclude_asin: str,
                          kws: list[str], n: int = 5) -> list[dict]:
        results = self._search(query, page=1)
        out: list[dict] = []
        for r in results:
            if r.get("asin") == exclude_asin: continue
            if not _matches(r, kws):           continue
            if float(r.get("rating", 0) or 0)  < CFG.min_comp_rating:  continue
            if float(r.get("reviews", 0) or 0) < CFG.min_comp_reviews: continue
            out.append(r)
            if len(out) >= n: break
        return out


INGESTER = AmazonIngester()


def ingest(mode: str, **kwargs) -> dict:
    """
    Public entrypoint.
    Returns: {"target": dict, "competitors": [...], "source": str, "comp_query": str, "kws": [...]}
    """
    if mode == "asin":
        asin = kwargs["asin"]
        comp_query = kwargs.get("comp_query")
        target = INGESTER.fetch_by_asin(asin)
        cat, kws = infer_category(target.get("title", ""))
        comp_query = comp_query or (cat or target.get("title", "")[:50])
        comps = INGESTER.fetch_competitors(comp_query, asin, kws, CFG.top_n_competitors)
        return {"target": target, "competitors": comps,
                "source": f"direct (ASIN {asin})", "comp_query": comp_query, "kws": kws}

    if mode == "url":
        asin = asin_from_url(kwargs["url"])
        if not asin:
            raise ValueError(f"Cannot parse ASIN from: {kwargs['url']}")
        return ingest("asin", asin=asin, comp_query=kwargs.get("comp_query"))

    if mode == "query":
        q = kwargs["query"]
        kws = infer_category(q)[1]
        target, page = INGESTER.fetch_buried(q, kws)
        comps = INGESTER.fetch_competitors(q, target.get("asin", ""), kws, CFG.top_n_competitors)
        return {"target": target, "competitors": comps,
                "source": f"page {page}", "comp_query": q, "kws": kws}

    raise ValueError(f"Unknown mode: {mode}")