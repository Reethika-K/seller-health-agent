"""
Diagnostic agent: takes a confirmed seller from the monitoring step and explains WHY it is declining.

  python diagnostic_agent.py 2026-09-08 --facts          no Gemini: only the numbers computed by code
  python diagnostic_agent.py 2026-09-08                  full run on the top 5 confirmed sellers
  python diagnostic_agent.py 2026-09-08 --top 3 --platform nykaa

How the work is split:
  CODE   decides the exact facts: onset date, sudden or gradual, whether the secondary category
         started after the primary one (a knock-on effect).
  LLM    looks at the seller's numbers with tools, searches the knowledge base (RAG), and writes
         the ranked likely causes with evidence. It cannot change the code's facts.
"""
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import diagnostic_tools as dt
import kb
import monitor_tools as mt

load_dotenv()
MODEL = "gemini-3.5-flash-lite"
TOP_N = 5                      # how many confirmed sellers to diagnose
MAX_TOOL_CALLS = 10
KNOCK_ON_GAP_DAYS = 3          # secondary must start at least this many days after the primary


def make_client():
    """Gemini client that fails after 90 seconds instead of waiting silently forever."""
    opts = {"timeout": 90_000}                       # milliseconds
    if hasattr(types, "HttpRetryOptions"):
        opts["retry_options"] = types.HttpRetryOptions(attempts=1)   # our own retry below prints messages
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "missing"),
                        http_options=types.HttpOptions(**opts))


client = make_client()
kb.client = client                                   # the knowledge-base search uses the same client


def _traced(fn):
    """Print every tool call the LLM makes, with how long it took."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        shown = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in kwargs.items())
        print(f"      tool {fn.__name__}({shown})", flush=True)
        try:
            return fn(*args, **kwargs)
        finally:
            print(f"      -> done in {time.time() - t0:.1f}s", flush=True)
    return wrapper


TRACED_TOOLS = [_traced(f) for f in dt.TOOLS]

SYSTEM = """You are the Diagnostic Agent in a seller-health system for an e-commerce marketplace.
A monitoring step has already found a declining seller and told you its primary category (the root
problem) and any secondary categories. Your job is to explain WHY the seller is declining, using
the seller's own numbers and the failure-pattern knowledge base. You do not change the primary or
secondary categories.

HOW TO WORK
1. Call get_seller_snapshot once. Note which metrics moved AND which did not move. Metrics that
   stayed flat are evidence too, for example flat stockouts make a stock problem unlikely.
2. Call get_category_timeline for the primary category, and for each clear secondary category.
   Movements in any other category are weak signals: small sellers have noisy averages. Do not use
   them as evidence or causes, and do not mention them as problems, unless a timeline call for that
   category shows a clear onset.
3. Call search_knowledge with category set to the primary category. Write the query as one or two
   plain sentences describing the pattern, for example "Orders are shipped late while order volume
   and stock levels are normal". Do not paste lists of metric names, and avoid negations like "flat"
   or "not rising" because the search matches on topic and not on meaning of "not". If the first
   results do not fit, search once more with different words.
4. Final answer in plain text, covering:
   - the 1 to 3 most likely causes, best first, each with a confidence (high, medium or low), the
     metric values from your tools that support it, and the knowledge document IDs you relied on;
   - for each secondary category, whether it looks like a knock-on effect of the primary one, using
     the onset dates from the timeline tool;
   - a 2 to 3 sentence summary a manager can read.
Only state numbers you got from a tool and only cite document IDs the search returned. If the
evidence does not support any cause well, say so and use low confidence."""


class RootCause(BaseModel):
    cause: str = Field(description="the likely cause in one sentence")
    confidence: Literal["high", "medium", "low"]
    evidence: List[str] = Field(description="metric values from the tools that support the cause")
    knowledge_docs: List[str] = Field(description="IDs of knowledge documents used, for example LOG-01")


class DiagnosisNarrative(BaseModel):
    summary: str = Field(description="2-3 sentences for a manager")
    root_causes: List[RootCause]
    secondary_assessment: str = Field(description="are the secondary categories knock-on effects, and why")


class Diagnosis(BaseModel):
    seller_id: str
    primary_category: str
    # decided by code
    secondary_categories: List[str]         # secondary categories with a clear, established signal
    weak_secondary_signals: List[str]       # flagged secondaries without one (probably noise)
    estimated_onset_date: Optional[str]
    decline_speed: str
    secondary_is_knock_on: Optional[bool]
    # written by the LLM
    summary: str
    root_causes: List[RootCause]
    secondary_assessment: str


def _generate(**kwargs):
    for attempt in range(6):
        try:
            return client.models.generate_content(model=MODEL, **kwargs)
        except Exception as e:
            msg = str(e)
            temporary = any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "504",
                                           "timed out", "Timeout", "DEADLINE"))
            if temporary and attempt < 5:
                wait = 30 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 15 * (attempt + 1)
                print(f"  Gemini slow or unavailable ({msg[:60]}), waiting {wait}s then retrying...", flush=True)
                time.sleep(wait)
            else:
                raise


def code_facts(cand, as_of):
    """The exact facts, computed without the LLM."""
    d = pd.Timestamp(as_of)
    primary = dt.category_facts(cand["seller_id"], cand["primary_category"], d)
    clear, weak, knock = [], [], None
    for sec in cand["secondary_categories"]:
        f = dt.category_facts(cand["seller_id"], sec, d)
        if not f["clear_signal"]:
            weak.append(sec)               # flagged, but no established onset: probably noise
            continue
        clear.append(sec)
        if f["estimated_onset_date"] and primary["estimated_onset_date"]:
            gap = (pd.Timestamp(f["estimated_onset_date"]) - pd.Timestamp(primary["estimated_onset_date"])).days
            knock = bool(knock) or gap >= KNOCK_ON_GAP_DAYS
    return {"estimated_onset_date": primary["estimated_onset_date"],
            "decline_speed": primary["decline_speed"],
            "secondary_categories": clear, "weak_secondary_signals": weak,
            "secondary_is_knock_on": knock}


def investigate(cand, as_of, facts):
    """Phase 1: the LLM calls the tools and searches the knowledge base."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=TRACED_TOOLS, temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=MAX_TOOL_CALLS))
    task = (f"Platform {mt.PLATFORM}, as of {as_of}. Diagnose seller {cand['seller_id']} "
            f"({cand['size_tier']} seller). Primary category: {cand['primary_category']} "
            f"(flagged in {cand['primary_consecutive_weeks']} recent weekly runs, severity {cand['severity']}). "
            f"Clear secondary categories: {facts['secondary_categories'] or 'none'}. "
            f"Weak signals (probably noise, do not build a cause on them): "
            f"{facts['weak_secondary_signals'] or 'none'}.")
    print("   asking Gemini to investigate...", flush=True)
    return _generate(contents=task, config=config).text


def structure(findings, seller_id):
    """Phase 2: turn the free-text findings into validated JSON."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=DiagnosisNarrative, temperature=0)
    prompt = (f"Convert the findings about seller {seller_id} into the schema. Do not add causes, "
              f"numbers or document IDs that are not in the findings.\n\nFINDINGS:\n{findings}")
    print("   structuring the findings...", flush=True)
    return _generate(contents=prompt, config=config).parsed


def diagnose(cand, as_of):
    facts = code_facts(cand, as_of)
    findings = investigate(cand, as_of, facts)
    n = structure(findings, cand["seller_id"])
    return Diagnosis(seller_id=cand["seller_id"], primary_category=cand["primary_category"], **facts,
                     summary=n.summary, root_causes=n.root_causes,
                     secondary_assessment=n.secondary_assessment)


def evaluate(items):
    """Compare the code-decided facts with the hidden ground truth (testing only).
    items = list of (candidate, facts)."""
    gt = pd.read_csv(f"{mt.PLATFORM}_data/ground_truth.csv", parse_dates=["decline_start_date"]).set_index("seller_id")
    errs, speed_ok, kn = [], [], []
    for cand, f in items:
        sid = cand["seller_id"]
        if gt.loc[sid, "is_at_risk"] != 1:
            continue
        if f["estimated_onset_date"]:
            errs.append((pd.Timestamp(f["estimated_onset_date"]) - gt.loc[sid, "decline_start_date"]).days)
        speed_ok.append(f["decline_speed"] == gt.loc[sid, "decline_speed"])
        true_sec = gt.loc[sid, "secondary_category"]
        if isinstance(true_sec, str) and true_sec in cand["secondary_categories"]:   # reported by monitoring
            kn.append(true_sec in f["secondary_categories"] and f["secondary_is_knock_on"] is True)
    print(f"\nEVALUATION as of {as_of_global}  ({len(speed_ok)} truly declining sellers diagnosed)")
    if errs:
        e = np.array(errs)
        print(f"Onset date: median error {np.median(e):.0f} days, within 7 days for {int((np.abs(e) <= 7).sum())}/{len(e)}")
    print(f"Sudden vs gradual correct:       {sum(speed_ok)}/{len(speed_ok)}")
    print(f"Real knock-on judged knock-on:   {sum(kn)}/{len(kn)}")


if __name__ == "__main__":
    as_of_global = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "2026-09-08"
    if "--platform" in sys.argv:
        mt.set_platform(sys.argv[sys.argv.index("--platform") + 1])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else TOP_N
    cands = [c for c in mt.get_flagged_candidates(as_of_global)["candidates"] if c["status"] == "confirmed"][:top]
    print(f"Diagnosing {len(cands)} confirmed sellers on {mt.PLATFORM}, as of {as_of_global}\n")

    facts_only = "--facts" in sys.argv
    results, evaluated = [], []
    for cand in cands:
        facts = code_facts(cand, as_of_global)
        evaluated.append((cand, facts))
        print(f"{cand['seller_id']}  {cand['primary_category']:20s} secondary={facts['secondary_categories']} "
              f"weak={facts['weak_secondary_signals']}")
        print(f"   onset {facts['estimated_onset_date']}  speed {facts['decline_speed']}  "
              f"secondary knock-on: {facts['secondary_is_knock_on']}")
        if facts_only:
            continue
        d = diagnose(cand, as_of_global)
        results.append(d)
        print(f"   {d.summary}")
        for rc in d.root_causes:
            print(f"   - [{rc.confidence}] {rc.cause}  ({', '.join(rc.knowledge_docs)})")
        time.sleep(3)                                  # be gentle with the free-tier rate limit
    if results:
        Path("reports").mkdir(exist_ok=True)
        out = Path("reports") / f"diagnosis_{mt.PLATFORM}_{as_of_global}.json"
        out.write_text(json.dumps([r.model_dump() for r in results], indent=2))
        print(f"\nSaved {out}")
    evaluate(evaluated)