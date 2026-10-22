"""
Orchestrator: runs the whole Seller Health pipeline for one date.

    monitor  ->  select  ->  diagnose  ->  draft  ->  finalize

  python orchestrator.py 2026-09-08                  run everything (top 5 confirmed sellers)
  python orchestrator.py 2026-09-08 --top 3          fewer sellers
  python orchestrator.py 2026-09-08 --fresh          ignore saved results and redo every step
  python orchestrator.py --graph                     print the pipeline as a Mermaid diagram

It is built with LangGraph. Each step is a node, the shared STATE carries results from one step
to the next, and two conditional edges stop early when there is nothing left to do:
  select    -> finalize  if no seller is confirmed
  diagnose  -> finalize  if every diagnosis failed

Resilience: results of each step are saved in reports/, and a rerun reuses them (use --fresh to
redo). If one seller fails, the pipeline records the failure and carries on with the others. If
the LLM part of monitoring fails, it falls back to the rule-based report.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

import diagnostic_agent as da
import drafting_agent as dg
import monitor_agent as ma
import monitor_tools as mt

REPORTS = Path("reports")
PAUSE = 3          # seconds between sellers, to be gentle with the free-tier rate limit


class PipelineState(TypedDict, total=False):
    as_of_date: str
    top_n: int
    fresh: bool
    monitor_report: dict       # written by monitor
    to_diagnose: List[dict]    # chosen by select (rule-based candidates)
    diagnoses: List[dict]      # written by diagnose
    plans: List[dict]          # written by draft
    failures: List[str]        # every problem met along the way


def _path(kind, date):
    REPORTS.mkdir(exist_ok=True)
    return REPORTS / f"{kind}_{mt.PLATFORM}_{date}.json"


def _load(path):
    return json.loads(path.read_text()) if path.exists() else []


# ------------------------------------------------------------------ nodes
def monitor_node(state):
    date, failures = state["as_of_date"], list(state.get("failures", []))
    path = _path("monitor", date)
    if path.exists() and not state.get("fresh"):
        print(f"[monitor] reusing {path}")
        return {"monitor_report": json.loads(path.read_text()), "failures": failures}
    print("[monitor] investigating with Gemini...", flush=True)
    narrative = None
    try:
        narrative = ma.structure(ma.investigate(date))
    except Exception as e:
        failures.append(f"monitor LLM step failed, used rules only: {str(e)[:120]}")
    if narrative is None:
        print("[monitor] LLM step unavailable, using the rule-based report")
        narrative = ma.MonitoringNarrative(summary="Rule-based report (the LLM step was unavailable).", sellers=[])
    report = ma.assemble_report(date, narrative).model_dump()
    path.write_text(json.dumps(report, indent=2))
    return {"monitor_report": report, "failures": failures}


def select_node(state):
    date = state["as_of_date"]
    cands = {c["seller_id"]: c for c in mt.get_flagged_candidates(date)["candidates"]}
    confirmed = [a["seller_id"] for a in state["monitor_report"]["alerts"] if a["status"] == "confirmed"]
    chosen = [cands[s] for s in confirmed if s in cands][:state.get("top_n", da.TOP_N)]
    print(f"[select] {len(confirmed)} confirmed sellers, diagnosing the top {len(chosen)}")
    return {"to_diagnose": chosen}


def diagnose_node(state):
    date, failures = state["as_of_date"], list(state.get("failures", []))
    path = _path("diagnosis", date)
    saved = {d["seller_id"]: d for d in _load(path)}
    result = []
    for cand in state["to_diagnose"]:
        sid = cand["seller_id"]
        if sid in saved and not state.get("fresh"):
            print(f"[diagnose] {sid}: reusing saved diagnosis")
            result.append(saved[sid])
            continue
        print(f"[diagnose] {sid} ({cand['primary_category']})...", flush=True)
        try:
            saved[sid] = da.diagnose(cand, date).model_dump()
            result.append(saved[sid])
            path.write_text(json.dumps(list(saved.values()), indent=2))
        except Exception as e:
            failures.append(f"diagnosis failed for {sid}: {str(e)[:120]}")
            print(f"[diagnose] {sid} failed, continuing")
        time.sleep(PAUSE)
    return {"diagnoses": result, "failures": failures}


def draft_node(state):
    date, failures = state["as_of_date"], list(state.get("failures", []))
    path = _path("actions", date)
    saved = {p["seller_id"]: p for p in _load(path)}
    cands = {c["seller_id"]: c for c in state["to_diagnose"]}
    result = []
    for diag in state["diagnoses"]:
        sid = diag["seller_id"]
        if sid in saved and not state.get("fresh"):
            print(f"[draft] {sid}: reusing saved plan")
            result.append(saved[sid])
            continue
        print(f"[draft] {sid}...", flush=True)
        try:
            saved[sid] = dg.draft(diag, cands[sid], date).model_dump()
            result.append(saved[sid])
            path.write_text(json.dumps(list(saved.values()), indent=2))
        except Exception as e:
            failures.append(f"drafting failed for {sid}: {str(e)[:120]}")
            print(f"[draft] {sid} failed, continuing")
        time.sleep(PAUSE)
    return {"plans": result, "failures": failures}


def finalize_node(state):
    date = state["as_of_date"]
    plans = sorted(state.get("plans", []), key=lambda p: (p["priority"], p["seller_id"]))
    # the headline numbers come from code: the LLM only saw the few sellers it investigated
    counts = mt.get_flagged_candidates(date)
    areas = Counter(a["primary_category"] for a in state["monitor_report"]["alerts"] if a["status"] == "confirmed")
    headline = (f"{counts['confirmed']} confirmed declining sellers and {counts['watch']} on watch on "
                f"{mt.PLATFORM} as of {date}. Main problem areas among confirmed sellers: "
                + (", ".join(f"{c} ({n})" for c, n in areas.most_common()) or "none") + ".")
    summary = {"platform": mt.PLATFORM, "as_of_date": date,
               "monitoring_summary": headline,
               "monitoring_notes_from_llm": state["monitor_report"]["summary"],
               "confirmed_sellers": counts["confirmed"],
               "diagnosed": len(state.get("diagnoses", [])), "drafted": len(plans),
               "failures": state.get("failures", []), "action_plans": plans}
    _path("pipeline", date).write_text(json.dumps(summary, indent=2))
    print(f"\n=== Pipeline finished: {mt.PLATFORM}, {date} ===")
    print(headline)
    print(f"Confirmed sellers: {summary['confirmed_sellers']} | diagnosed: {summary['diagnosed']} | drafted: {summary['drafted']}")
    for p in plans:
        print(f"  {p['priority']}  {p['seller_id']}  {p['primary_category']:20s} {p['owner_team']:26s}"
              f" onset {p['estimated_onset_date']} ({p['decline_speed']})")
    if summary["failures"]:
        print("Problems:")
        for f in summary["failures"]:
            print("  -", f)
    print(f"Saved reports/pipeline_{mt.PLATFORM}_{date}.json")
    try:
        ma.evaluate(ma.MonitoringReport(**state["monitor_report"]))
    except Exception:
        pass
    return {}


# ------------------------------------------------------------------ routing and graph
def after_select(state):
    return "diagnose" if state["to_diagnose"] else "finalize"


def after_diagnose(state):
    return "draft" if state["diagnoses"] else "finalize"


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("monitor", monitor_node)
    g.add_node("select", select_node)
    g.add_node("diagnose", diagnose_node)
    g.add_node("draft", draft_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "monitor")
    g.add_edge("monitor", "select")
    g.add_conditional_edges("select", after_select, {"diagnose": "diagnose", "finalize": "finalize"})
    g.add_conditional_edges("diagnose", after_diagnose, {"draft": "draft", "finalize": "finalize"})
    g.add_edge("draft", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    if "--graph" in sys.argv:
        print(app.get_graph().draw_mermaid())
        sys.exit()
    if "--platform" in sys.argv:
        mt.set_platform(sys.argv[sys.argv.index("--platform") + 1])
    date = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "2026-09-08"
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else da.TOP_N
    app.invoke({"as_of_date": date, "top_n": top, "fresh": "--fresh" in sys.argv, "failures": []})