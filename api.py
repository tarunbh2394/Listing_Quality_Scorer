"""
REST API — your frontend calls this.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    POST /analyze        Body: {"mode":"asin","asin":"B0G2MM7VH9","comp_query":"laptop under 50000"}
    GET  /healthz        liveness check
"""
import logging
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Ingestion  import ingest
from Extraction import extract_all
from QC         import run_qc
from features   import build_all_features
from pipeline   import run_pipeline
from insights   import generate_insights

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Opsell LQS API", version="1.0.0")

# Allow your frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    mode      : Literal["asin", "url", "query"]
    asin      : Optional[str] = None
    url       : Optional[str] = None
    query     : Optional[str] = None
    comp_query: Optional[str] = None
    chart     : bool          = False   # set True to save chart


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run the full 6-step pipeline and return all insights as JSON."""
    try:
        kwargs = {}
        if req.asin:       kwargs["asin"]       = req.asin
        if req.url:        kwargs["url"]        = req.url
        if req.query:      kwargs["query"]      = req.query
        if req.comp_query: kwargs["comp_query"] = req.comp_query

        ingested  = ingest(req.mode, **kwargs)
        extracted = extract_all(ingested)
        qc_data   = run_qc(extracted)
        feat_data = build_all_features(qc_data)
        df        = run_pipeline(feat_data)
        insights  = generate_insights(df, extracted["source"], make_chart=req.chart)

        return {
            "ok"     : True,
            "source" : extracted["source"],
            "scores" : df.reset_index().rename(columns={"index": "label"}).to_dict("records"),
            **insights,
        }
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Unhandled error")
        raise HTTPException(status_code=500, detail=str(e))