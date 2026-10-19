"""
LLM monitoring agent.
Code decides WHO is declining (rules in monitor_tools.py: status, primary category, severity).
The LLM does what code can't: it investigates the most serious sellers with tools, gathers
evidence, and explains what is going on in plain language.

Phase 1 (investigate): Gemini calls the tools and writes up its findings.
Phase 2 (structure):   a second call turns those findings into validated JSON.
Then the report is assembled in code so every confirmed seller is always included.

Run:  python monitor_agent.py 2026-09-08          (calls Gemini)
      python monitor_agent.py 2026-09-08 --rules  (no LLM: just the rule-based list)
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

import monitor_tools as tools

load_dotenv()
MODEL = "gemini-3.5-flash-lite"
MAX_TOOL_CALLS = 20
TOP_N_TO_INVESTIGATE = 6
MAX_WATCH = 10
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "missing"))

SYSTEM = f"""You are the Monitoring Agent in a seller-health system for an e-commerce marketplace.
A scoring tool has already decided which sellers are declining. Your job is to investigate the
most serious ones, gather concrete evidence, and explain what is happening so the right team
can act. You never calculate scores yourself, and you never change a seller's status, primary
category, secondary categories or severity: those come from the tool.

HOW SCORES WORK
- Each seller has a sub-score per category (Marketing, Logistics, Customer Experience,
  Seller support, Finance, Inventory). 1.0 = no deterioration against the seller's own recent
  past; lower is worse; 0.35 or below is "flagged"; about 0.09 is critical.
- status "confirmed" = flagged for 2+ consecutive weeks. "watch" = flagged only this week.
- Secondary categories are usually knock-on effects of the primary one.

HOW TO WORK
1. Call get_flagged_candidates once for the date you are given.
2. For the first {TOP_N_TO_INVESTIGATE} confirmed candidates (the list is already sorted, most
   serious first) call get_metric_detail for the primary category, and for one secondary
   category if there is one. Use get_seller_history only if you need to see how a problem evolved.
3. Final answer, in plain text:
   - For each seller you investigated: seller_id, evidence (actual metric values from the tool,
     such as "late_shipment_rate 0.31 vs 0.07 before"), and 1-2 sentences of reasoning: what
     changed, and whether the secondary category looks like a knock-on effect.
   - Then a 2-3 sentence platform summary with the counts of confirmed and watch sellers and
     which categories dominate.
   Only state numbers you got from a tool. Do not invent sellers or categories.
"""


class SellerNarrative(BaseModel):
    seller_id: str
    evidence: List[str] = Field(description="metric values seen in tool results")
    reasoning: str


class MonitoringNarrative(BaseModel):
    summary: str = Field(description="2-3 sentences for a human reader")
    sellers: List[SellerNarrative]


class SellerAlert(BaseModel):
    seller_id: str
    status: str                      # from the rule: confirmed / watch
    severity: str                    # from the rule
    primary_category: str            # from the rule
    secondary_categories: List[str]  # from the rule
    evidence: List[str]
    reasoning: str
    investigated: bool = Field(description="True if the LLM looked at this seller with tools")


class MonitoringReport(BaseModel):
    platform: str
    as_of_date: str
    summary: str
    alerts: List[SellerAlert]


def _generate(**kwargs):
    """Call Gemini, waiting and retrying if the free tier says we are going too fast."""
    for attempt in range(4):
        try:
            return client.models.generate_content(model=MODEL, **kwargs)
        except Exception as e:
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                print(f"  rate limited, waiting 30s (attempt {attempt + 1})...")
                time.sleep(30)
            else:
                raise


def investigate(as_of_date):
    """Phase 1: the LLM explores the data by calling tools."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=tools.TOOLS, temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=MAX_TOOL_CALLS))
    task = (f"Assess the {tools.PLATFORM} platform as of {as_of_date}. "
            f"Investigate the most serious declining sellers.")
    return _generate(contents=task, config=config).text


def structure(findings):
    """Phase 2: turn the free-text findings into validated JSON."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=MonitoringNarrative, temperature=0)
    prompt = ("Convert the findings below into the schema. Include only sellers that were "
              "investigated in the findings. Do not add sellers, numbers or facts that are "
              f"not in the findings.\n\nFINDINGS:\n{findings}")
    return _generate(contents=prompt, config=config).parsed


def assemble_report(as_of_date, narrative):
    """Combine the rule-based decisions with the LLM's evidence and reasoning.
    Every confirmed seller is included, whether or not the LLM investigated it."""
    candidates = tools.get_flagged_candidates(as_of_date)["candidates"]
    notes = {n.seller_id: n for n in narrative.sellers}
    alerts, watch_shown = [], 0
    for c in candidates:
        if c["status"] == "watch":
            if watch_shown >= MAX_WATCH:
                continue
            watch_shown += 1
        n = notes.get(c["seller_id"])
        evidence = n.evidence if n else [
            f"{c['primary_category']} sub-score {c['primary_score']} "
            f"(flagged {c['primary_consecutive_weeks']} week(s) in a row)"]
        reasoning = n.reasoning if n else "Not investigated in detail (lower priority); the status comes from the scoring rule."
        alerts.append(SellerAlert(
            seller_id=c["seller_id"], status=c["status"], severity=c["severity"],
            primary_category=c["primary_category"],
            secondary_categories=list(c["secondary_categories"]),
            evidence=evidence, reasoning=reasoning, investigated=n is not None))
    return MonitoringReport(platform=tools.PLATFORM, as_of_date=as_of_date,
                            summary=narrative.summary, alerts=alerts)


def evaluate(report):
    """Check the report against the hidden ground truth (for testing only)."""
    gt = pd.read_csv(f"{tools.PLATFORM}_data/ground_truth.csv").set_index("seller_id")
    risk = set(gt.index[gt.is_at_risk == 1])
    confirmed = [a for a in report.alerts if a.status == "confirmed"]
    true_pos = [a for a in confirmed if a.seller_id in risk]
    routed = sum(a.primary_category in (gt.loc[a.seller_id, "primary_category"],
                                        gt.loc[a.seller_id, "secondary_category"]) for a in true_pos)
    print(f"\nEVALUATION as of {report.as_of_date}")
    print(f"Confirmed sellers in the report: {len(confirmed)}")
    print(f"Truly declining among them:      {len(true_pos)}/{len(confirmed)}")
    print(f"Primary category correct:        {routed}/{len(true_pos)}")
    print(f"Investigated by the LLM with tools: {sum(a.investigated for a in report.alerts)} sellers")


if __name__ == "__main__":
    as_of = sys.argv[1] if len(sys.argv) > 1 else "2026-09-08"
    if "--rules" in sys.argv:
        r = tools.get_flagged_candidates(as_of)
        print(f"{r['confirmed']} confirmed, {r['watch']} watch")
        for c in r["candidates"]:
            if c["status"] == "confirmed":
                print(c["seller_id"], c["severity"], c["primary_category"], list(c["secondary_categories"]))
        sys.exit()

    print("Phase 1: investigating with tools...")
    findings = investigate(as_of)
    print(findings, "\n")
    print("Phase 2: structuring the findings...")
    narrative = structure(findings)
    report = assemble_report(as_of, narrative)
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports") / f"monitor_{tools.PLATFORM}_{as_of}.json"
    out.write_text(report.model_dump_json(indent=2))
    print(f"Saved {out}\n{report.summary}")
    evaluate(report)