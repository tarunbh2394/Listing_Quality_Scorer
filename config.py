"""Global configuration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    # API
    api_key      : str  = "ok_7a7f092c1855099ab70464867575fc29"
    base_url     : str  = "https://amazon-scraper-api.omkar.cloud"
    country_code : str  = "IN"
    timeout_s    : int  = 60

    # Model
    model_path   : Path = Path(r"C:\Users\tarun\OneDrive\Desktop\Opsell\LQS_Project\models\lqs_model.pkl")

    # Scoring
    lqs_min       : float = 65.0
    lqs_max       : float = 95.0
    grade_a_cut   : float = 88.0
    grade_b_cut   : float = 75.0
    blend_model_w : float = 0.4

    # Competitor selection
    top_n_competitors : int             = 5
    buried_pages      : tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    min_comp_rating   : float           = 0.0
    min_comp_reviews  : int             = 0

    @property
    def headers(self) -> dict[str, str]:
        return {"API-Key": self.api_key}


CFG = Config()

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "laptop"   : ["laptop","notebook","chromebook","macbook","vivobook","ideapad",
                  "inspiron","pavilion","aspire","zenbook","thinkpad","elitebook",
                  "probook","gram","swift","ryzen","core i3","core i5","core i7"],
    "phone"    : ["phone","smartphone","iphone","galaxy","redmi","poco","oneplus",
                  "realme","vivo","oppo","pixel","mobile"],
    "headphone": ["headphone","earphone","earbuds","tws","airpods","headset","neckband"],
    "tv"       : ["tv","television","smart tv","qled","oled","led tv"],
    "tablet"   : ["tablet","ipad","fire hd"],
}

BRAND_VOCAB: set[str] = {
    "hp","dell","lenovo","asus","acer","apple","samsung","msi","lg","sony",
    "xiaomi","realme","oneplus","redmi","infinix","boat","jbl","sennheiser",
    "bose","noise","mi","iphone","macbook","thomson","toshiba","huawei",
}