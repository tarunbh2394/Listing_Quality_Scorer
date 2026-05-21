"""Opsell LQS — MCP server. Exposes Amazon LQS scoring tools callable by any LLM."""
import math, re
import joblib, numpy as np, pandas as pd, requests
from mcp.server.fastmcp import FastMCP

API_KEY    = "ok_7a7f092c1855099ab70464867575fc29"
BASE_URL   = "https://amazon-scraper-api.omkar.cloud"
COUNTRY    = "IN"
HEADERS    = {"API-Key": API_KEY}
MODEL_PATH = r"C:\Users\tarun\OneDrive\Desktop\Opsell\LQS_Project\models\lqs_model.pkl"

mcp = FastMCP("opsell-lqs")
MODEL = joblib.load(MODEL_PATH)
COLS = list(MODEL.feature_names_in_)
BRANDS = {"hp","dell","lenovo","asus","acer","apple","samsung","sony","bose","boat",
          "jbl","noise","mamaearth","wow","plum","ohria","lakshya","himalaya","dabur",
          "patanjali","mi","xiaomi","realme","oneplus","redmi"}
ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})")

def _asin(t):
    m = ASIN_RE.search(t or ""); return m.group(1) if m else t.strip()

def _fetch(q, page=1):
    r = requests.get(f"{BASE_URL}/amazon/search",
                     params={"query":q,"country_code":COUNTRY,"page":page},
                     headers=HEADERS, timeout=60)
    r.raise_for_status(); return r.json().get("results", [])

def _by_asin(a):
    for r in _fetch(a):
        if str(r.get("asin","")).upper()==a.upper(): return r
    res=_fetch(a)
    if res: return res[0]
    raise ValueError(f"ASIN {a} not found")

def _features(p):
    title=str(p.get("title","") or ""); price=float(p.get("price",0) or 0)
    mrp=float(p.get("original_price",0) or 0); rating=float(p.get("rating",0) or 0)
    rev=float(p.get("reviews",0) or 0); disc=round((mrp-price)/mrp*100,2) if mrp>0 else 0
    tl=title.lower(); brand=1 if any(b in tl for b in BRANDS) else 0; est=rating>=4.0 and rev>=100
    return {"title_length":len(title),"title_word_count":len(title.split()),
        "title_brand_present":brand,"title_sentence_case":1 if title and title[0].isupper() else 0,
        "title_spellcheck_pass":1,"bullet_count":5 if est else 2,"bullets_spell_pass":1,
        "bullets_caps_pass":1,"final_bullet_flag":1 if est else 0,
        "description_word_count":200 if est else 40,"final_desc_flag":1 if est else 0,
        "price":price,"mrp":mrp,"discount_pct":disc,"star_rating":rating,"review_count":rev,
        "log_review_count":np.log1p(rev),"best_seller_rank":max(1,int(50000/(rev+10))),
        "image_count":7 if est else 3,"video_count":1 if (brand and rev>=500) else 0,
        "has_video":1 if (brand and rev>=500) else 0,"has_aplus":1 if (brand and rev>=100) else 0,
        "aplus_check_pass":1 if (brand and rev>=100) else 0}

def _heur(f):
    s=min(f["star_rating"]/5,1)*30+min(np.log1p(f["review_count"])/np.log1p(5000),1)*30
    s+=10 if f["title_brand_present"] else 0
    s+=10 if f["discount_pct"]>10 else (5 if f["discount_pct"]>0 else 0)
    s+=10 if 80<=f["title_length"]<=200 else (5 if f["title_length"]>=50 else 0)
    s+=5 if f["image_count"]>=1 else 0
    s+=5 if (f["star_rating"]>=4 and f["review_count"]>=100) else 0
    return s

def _score(p):
    f=_features(p); X=pd.DataFrame([{c:f.get(c,0) for c in COLS}])[COLS]
    ms=float(np.clip(MODEL.predict(X)[0],65,95)); hs=65+(_heur(f)/100)*30
    lqs=float(np.clip(0.4*ms+0.6*hs,65,95)); grade="A" if lqs>=88 else ("B" if lqs>=75 else "C")
    asp=f["price"] if f["price"]>0 else 1
    ctr=round(min((min(f["star_rating"]/5,1)*30+min(math.log1p(f["review_count"])/math.log1p(5000),1)*25
        +(10 if f["discount_pct"]>10 else 0)+(10 if f["image_count"]>=1 else 0)
        +(10 if p.get("is_prime") else 0))/100,1),3)
    cvr=round(min((min(f["star_rating"]/5,1)*35+min(math.log1p(f["review_count"])/math.log1p(5000),1)*30
        +(15 if p.get("is_prime") else 0)+(10 if f["discount_pct"]>10 else 0))/100,1),3)
    return {"title":str(p.get("title",""))[:80],"asin":p.get("asin","N/A"),"price":asp,
        "discount_pct":f["discount_pct"],"rating":f["star_rating"],"reviews":int(f["review_count"]),
        "lqs":round(lqs,1),"grade":grade,"ctr":ctr,"cvr":cvr,"rpi":round(asp*ctr*cvr,2)}

@mcp.tool()
def score_amazon_listing(asin_or_url: str) -> dict:
    """Score an Amazon listing's quality. Pass a 10-char ASIN (e.g. B0G2MM7VH9) or product URL.
    Returns LQS (65-95), grade A/B/C, estimated CTR, CVR, and RPI (price x CTR x CVR)."""
    return _score(_by_asin(_asin(asin_or_url)))

@mcp.tool()
def compare_with_competitors(asin_or_url: str, competitor_query: str, top_n: int = 5) -> dict:
    """Score an Amazon listing and compare it to its top competitors.
    competitor_query is a search phrase like 'laptop under 50000'.
    Returns target scores, competitor scores, and the gap vs the best competitor."""
    a=_asin(asin_or_url); target=_score(_by_asin(a))
    comps=[r for r in _fetch(competitor_query,1) if r.get("asin")!=a][:top_n]
    cs=[_score(c) for c in comps]
    best=max(cs,key=lambda x:x["rpi"]) if cs else None
    gap={m:round(best[m]-target[m],3) for m in ["lqs","ctr","cvr","rpi","rating","reviews"]} if best else {}
    return {"target":target,"competitors":cs,"best_competitor":best,"gap_vs_best":gap}

if __name__ == "__main__":
    mcp.run()
