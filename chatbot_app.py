"""
Seller Health Monitor - ASSISTANT ONLY (for testing the chatbot).

    streamlit run chatbot_app.py          (run it from the project folder)

Same sidebar as app.py (platform, as-of date, run pipeline, reload), but the page is just the chatbot.
It reads the reports in reports/ and the daily data through read-only tools. It cannot run the pipeline
by chat or send anything. app.py is unchanged.
"""
import contextlib
import functools
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
os.chdir(BASE)                                   # the agents use relative paths (reports/, <platform>_data/)

st.set_page_config(page_title="Seller Health Assistant (test)", layout="wide", initial_sidebar_state="expanded")

import monitor_tools as mt                       # noqa: E402
from ahp_config import CATEGORIES, METRICS, PLATFORM_MATRICES   # noqa: E402
from levels import HIGHER_IS_BETTER              # noqa: E402

# The agent modules need the Gemini packages and the knowledge-base index. If something is missing
# the app still shows the saved reports; only the Gemini features are switched off.
AGENT_ERROR = None
try:
    import diagnostic_tools as dt
    import drafting_agent as dg
    import kb
    kb.client = dg.client
except Exception as exc:                         # pragma: no cover
    dt = dg = kb = None
    AGENT_ERROR = f"{type(exc).__name__}: {exc}"

REPORTS = BASE / "reports"
MODEL = "gemini-3.5-flash-lite"
FALLBACK_PRIORITY = {"Critical": ("P1", 2), "At-risk": ("P2", 7)}
FALLBACK_OWNER = {"Marketing": "Catalog and ads team", "Logistics": "Logistics operations",
                  "Customer Experience": "Customer experience team", "Seller support": "Seller support team",
                  "Finance": "Finance operations", "Inventory": "Seller supply team"}
PRIORITY_RULE = dg.PRIORITY if dg else FALLBACK_PRIORITY
OWNER_TEAM = dg.OWNER_TEAM if dg else FALLBACK_OWNER
JARGON = dg.JARGON if dg else ["sub-score", "subscore", "ahp", "knowledge base", "diagnostic", "ground truth", "flagged"]
FLAG_SCORE = mt.FLAG_SCORE

BLUE, RED, GREY, AMBER = "#2563eb", "#dc2626", "#9ca3af", "#d97706"


# ======================================================================== data helpers
def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def use_platform(platform: str):
    if mt.PLATFORM != platform:
        mt.set_platform(platform)


def available_platforms():
    return [p for p in PLATFORM_MATRICES if (BASE / f"{p}_data" / "seller_daily_metrics.csv").exists()]


@st.cache_data(show_spinner=False)
def data_range(platform: str):
    df = pd.read_csv(BASE / f"{platform}_data" / "seller_daily_metrics.csv", usecols=["date"], parse_dates=["date"])
    return df.date.min(), df.date.max()


def weekly_dates(platform: str):
    """The dates the scoring can run on: weekly, starting 34 days after the data begins."""
    first, last = data_range(platform)
    d, out = first + pd.Timedelta(days=34), []
    while d <= last:
        out.append(str(d.date()))
        d += pd.Timedelta(days=7)
    return out


def saved_dates(platform: str):
    found = set()
    for f in REPORTS.glob(f"*_{platform}_*.json"):
        if f.stem.startswith("approved_"):
            continue
        found.add(f.stem.split("_")[-1])
    return found


@st.cache_data(show_spinner=False)
def sellers_table(platform: str):
    return pd.read_csv(BASE / f"{platform}_data" / "sellers.csv").set_index("seller_id")


@st.cache_data(show_spinner="Scoring every seller...")
def candidates(platform: str, date: str):
    use_platform(platform)
    return mt.get_flagged_candidates(date)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_reports(platform: str, date: str):
    """Everything the pipeline saved for this platform and date."""
    pipe = read_json(REPORTS / f"pipeline_{platform}_{date}.json", {})
    monitor = read_json(REPORTS / f"monitor_{platform}_{date}.json", {})
    diags = {d["seller_id"]: d for d in read_json(REPORTS / f"diagnosis_{platform}_{date}.json", [])}
    plans = {p["seller_id"]: p for p in pipe.get("action_plans", [])}
    plans.update({p["seller_id"]: p for p in read_json(REPORTS / f"actions_{platform}_{date}.json", [])})
    alerts = {a["seller_id"]: a for a in monitor.get("alerts", [])}
    return {"pipeline": pipe, "monitor": monitor, "diagnoses": diags, "plans": plans, "alerts": alerts}


def approvals_path(platform, date):
    return REPORTS / f"approved_{platform}_{date}.json"


def load_approvals(platform, date):
    return read_json(approvals_path(platform, date), {})


def save_approvals(platform, date, data):
    REPORTS.mkdir(exist_ok=True)
    approvals_path(platform, date).write_text(json.dumps(data, indent=2))


@st.cache_data(show_spinner=False)
def kb_titles():
    if not kb:
        return {}
    try:
        return {d["id"]: d for d in kb.load_documents()}
    except Exception:
        return {}


def priority_of(c, plan):
    """P1/P2 from the saved plan, or from the same severity rule the drafting step uses."""
    if c["status"] != "confirmed":
        return "Watch"
    return plan["priority"] if plan else PRIORITY_RULE[c["severity"]][0]


def message_status(sid, plan, approvals):
    if sid in approvals and approvals[sid].get("approved"):
        return "Approved"
    return "Draft ready" if plan else "-"


# ======================================================================== sidebar
def sidebar():
    st.sidebar.title("Seller Health Monitor")
    platforms = available_platforms()
    if not platforms:
        st.sidebar.error("No <platform>_data/ folder found next to app.py.")
        st.stop()
    platform = st.sidebar.selectbox("Platform", platforms, format_func=str.capitalize, key="platform")
    use_platform(platform)

    saved = saved_dates(platform)
    dates = weekly_dates(platform)
    default_date = max(saved & set(dates)) if saved & set(dates) else dates[-1]
    if "as_of" not in st.session_state or st.session_state["as_of"] not in dates:
        st.session_state["as_of"] = default_date
    date = st.sidebar.selectbox("As-of date", dates[::-1], key="as_of",
                                format_func=lambda d: d + ("   (saved report)" if d in saved else ""),
                                help="The app shows the platform as it looked on this date. "
                                     "Dates with a saved report also show the AI analysis.")

    top_n = st.sidebar.slider("Sellers to diagnose per run", 1, 10, 5,
                              help="How many of the most serious confirmed sellers the pipeline investigates.")

    st.sidebar.divider()
    with st.sidebar.expander("Run the pipeline", expanded=False):
        st.caption("Runs monitor, select, diagnose and draft for this date. It calls Gemini and uses your free-tier "
                   "quota; each seller takes roughly one to two minutes. Steps already saved are reused.")
        fresh = st.checkbox("Redo everything (ignore saved results)", value=False)
        if AGENT_ERROR:
            st.warning("The agents could not be loaded: " + AGENT_ERROR)
        elif not os.environ.get("GEMINI_API_KEY"):
            st.warning("GEMINI_API_KEY was not found. Add it to the .env file.")
        if st.button("Run pipeline", type="primary", disabled=bool(AGENT_ERROR), width="stretch"):
            run_pipeline(platform, date, top_n, fresh)

    if st.sidebar.button("Reload saved reports", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("Test app: only the assistant. Opening it never calls Gemini until you ask a question.")
    return platform, date, top_n


def run_pipeline(platform, date, top_n, fresh):
    import orchestrator as orch
    log = io.StringIO()
    names = {"monitor": "Monitoring", "select": "Selecting sellers", "diagnose": "Diagnosing",
             "draft": "Drafting actions", "finalize": "Summarising"}
    with st.status("Running the pipeline...", expanded=True) as status:
        try:
            graph = orch.build_graph()
            state = {"as_of_date": date, "top_n": top_n, "fresh": fresh, "failures": []}
            with contextlib.redirect_stdout(log):
                for update in graph.stream(state, stream_mode="updates"):
                    for node in update:
                        status.write(f"Finished: {names.get(node, node)}")
            status.update(label="Pipeline finished", state="complete", expanded=False)
        except Exception as exc:
            status.update(label=f"Pipeline stopped: {str(exc)[:150]}", state="error")
            st.session_state["run_log"] = log.getvalue()
            return
    st.session_state["run_log"] = log.getvalue()
    st.cache_data.clear()
    st.toast("Pipeline finished. Reports reloaded.")
    st.rerun()



def gemini_call(fn, attempts=4):
    """Call Gemini, waiting and retrying on temporary errors (rate limit, overload, timeout)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            msg = str(exc)
            temporary = any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "504",
                                               "timed out", "Timeout", "DEADLINE"))
            if temporary and i < attempts - 1:
                time.sleep(20 if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 6 * (i + 1))
            else:
                raise



# ======================================================================== assistant
TRACE = []


def traced(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        TRACE.append(f"{fn.__name__}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
        return fn(*args, **kwargs)
    return wrapper


def make_chat_tools(platform, date):
    """Read-only tools for the chatbot. The date and platform are fixed by the sidebar."""
    use_platform(platform)

    def list_sellers(status: str = "confirmed", category: str = "", limit: int = 10) -> dict:
        """List flagged sellers, most serious first.

        Args:
            status: "confirmed" (declining for 2+ weeks), "watch" (flagged this week only) or "all".
            category: optional main problem category, for example "Marketing" or "Logistics". Empty means all.
            limit: how many sellers to return (maximum 25).
        """
        cands = candidates(platform, date)["candidates"]
        reports = load_reports(platform, date)
        rows = []
        for c in cands:
            if status != "all" and c["status"] != status:
                continue
            if category and c["primary_category"].lower() != category.lower():
                continue
            rows.append({"seller_id": c["seller_id"], "priority": priority_of(c, reports["plans"].get(c["seller_id"])),
                         "status": c["status"], "main_problem": c["primary_category"],
                         "also_affected": list(c["secondary_categories"]), "weeks_flagged": c["primary_consecutive_weeks"],
                         "overall_score": c["overall_score"], "diagnosed": c["seller_id"] in reports["diagnoses"]})
        return {"as_of_date": date, "matching_sellers": len(rows), "sellers": rows[:min(limit, 25)]}

    def get_seller_report(seller_id: str) -> dict:
        """The saved analysis for one seller: monitoring reasoning, diagnosis (causes with evidence),
        the action plan and the draft message. Fields are missing when the seller was not diagnosed.

        Args:
            seller_id: for example "S0065".
        """
        reports = load_reports(platform, date)
        c = next((x for x in candidates(platform, date)["candidates"] if x["seller_id"] == seller_id), None)
        if c is None:
            return {"error": f"{seller_id} is not flagged on {date}"}
        out = {"seller_id": seller_id, "status": c["status"], "severity": c["severity"],
               "main_problem": c["primary_category"], "also_affected": c["secondary_categories"],
               "weeks_flagged": c["primary_consecutive_weeks"], "as_of_date": date}
        a, d, p = reports["alerts"].get(seller_id), reports["diagnoses"].get(seller_id), reports["plans"].get(seller_id)
        if a:
            out["monitoring"] = {"evidence": a["evidence"], "reasoning": a["reasoning"]}
        if d:
            out["diagnosis"] = {k: d[k] for k in ("summary", "root_causes", "secondary_assessment", "estimated_onset_date",
                                                  "decline_speed", "secondary_is_knock_on")}
        if p:
            out["action_plan"] = {k: p[k] for k in ("priority", "owner_team", "first_contact_within_days", "internal_summary",
                                                    "actions", "seller_message", "metrics_to_recheck", "escalate_if")}
        if not (d or p):
            out["note"] = "This seller has not been diagnosed, so there is no saved diagnosis or plan."
        return out

    def get_seller_snapshot(seller_id: str) -> dict:
        """All metrics of one seller for the last 7 days versus the 21 days before, with their levels.

        Args:
            seller_id: for example "S0065".
        """
        return dt.get_seller_snapshot(seller_id, date)

    def get_category_timeline(seller_id: str, category: str) -> dict:
        """When a category started getting worse for one seller, how fast, and which metrics moved.

        Args:
            seller_id: for example "S0065".
            category: Marketing, Logistics, Customer Experience, Seller support, Finance or Inventory.
        """
        return dt.get_category_timeline(seller_id, category, date)

    def get_seller_history(seller_id: str, weeks: int = 4) -> dict:
        """Category scores of one seller over the last weekly runs (1.0 healthy, lower is worse, 0.35 or less is flagged).

        Args:
            seller_id: for example "S0065".
            weeks: how many weekly runs, most recent first (maximum 6).
        """
        return mt.get_seller_history(seller_id, date, weeks)

    def search_knowledge(query: str, category: str = "", top_k: int = 3) -> list:
        """Search the failure-pattern knowledge base for documents explaining a symptom.

        Args:
            query: the symptom in plain words, naming metrics and their direction.
            category: optional category filter.
            top_k: number of documents to return.
        """
        return kb.search_knowledge(query, category=category, top_k=top_k)

    return [traced(f) for f in (list_sellers, get_seller_report, get_seller_snapshot, get_category_timeline,
                                get_seller_history, search_knowledge)]


def chat_system(platform, date):
    return f"""You are the assistant inside a Seller Health Monitor for the {platform.capitalize()} marketplace. The data is as of {date}.
Account managers ask you about declining sellers.

RULES
- Answer only from tool results. Never guess numbers, dates, causes or seller details. If the tools do not have it, say so.
- Seller IDs look like S0065. For any question about a particular seller call get_seller_report first; use the other tools for
  more evidence. For lists or counts call list_sellers.
- Category scores run from 1.0 (healthy) down to about 0.09; 0.35 or lower means the category is flagged. A seller is
  "confirmed" when a category was flagged in 2 or more weekly runs, otherwise "watch".
- You are read-only. You cannot run the pipeline, change anything, or send messages to sellers. If asked to, say that the user does
  it with "Run the pipeline" in the sidebar.
- If asked to rewrite a seller message, write it in your reply: polite, specific, under 150 words, and never mention scores, internal
  tools, document IDs or the words flagged, sub-score, AHP, diagnostic, knowledge base.
- Keep answers short and in plain language. Mention the seller ID and the date when you give facts."""


def new_chat(platform, date):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=chat_system(platform, date), tools=make_chat_tools(platform, date), temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8))
    return dg.client.chats.create(model=MODEL, config=cfg)


def view_assistant(platform, date, cands, reports):
    st.subheader("Ask about your sellers")
    st.caption("Answers come from the daily data and the saved reports, through read-only tools. "
               "The assistant cannot run the pipeline or send anything.")
    if AGENT_ERROR:
        st.error("The assistant needs the Gemini packages: " + AGENT_ERROR)
        return
    if not os.environ.get("GEMINI_API_KEY"):
        st.warning("GEMINI_API_KEY was not found in .env, so the assistant cannot answer.")

    key = (platform, date)
    if st.session_state.get("chat_key") != key:
        st.session_state["chat_key"] = key
        st.session_state["chat_msgs"] = []
        st.session_state.pop("chat", None)
    msgs = st.session_state["chat_msgs"]

    top = next((c["seller_id"] for c in cands["candidates"] if c["status"] == "confirmed"), None)
    suggestions = ["Which sellers need attention first?", "How many confirmed sellers are there per problem area?"]
    if top:
        suggestions += [f"Why is {top} declining, and what should we do?", f"Rewrite the message to {top} so it sounds friendlier."]
    pending = None
    if not msgs:
        cols = st.columns(len(suggestions))
        for col, s in zip(cols, suggestions):
            if col.button(s, width="stretch"):
                pending = s

    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                with st.expander("Tools used"):
                    for t in m["tools"]:
                        st.code(t, language="python")

    prompt = st.chat_input("Ask a question, for example: which Logistics sellers are P1?") or pending
    if msgs and st.button("Clear conversation"):
        st.session_state["chat_msgs"] = []
        st.session_state.pop("chat", None)
        st.rerun()
    if not prompt:
        return

    msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Looking it up..."):
            TRACE.clear()
            try:
                if "chat" not in st.session_state:
                    st.session_state["chat"] = new_chat(platform, date)
                resp = gemini_call(lambda: st.session_state["chat"].send_message(prompt))
                answer = resp.text or "I could not produce an answer. Try asking in a different way."
            except Exception as exc:
                answer = f"Sorry, the request failed: {str(exc)[:200]}. Try again in a moment."
        st.markdown(answer)
        used = list(TRACE)
        if used:
            with st.expander("Tools used"):
                for t in used:
                    st.code(t, language="python")
    msgs.append({"role": "assistant", "content": answer, "tools": used})



# ======================================================================== main
def main():
    platform, date, _top_n = sidebar()
    cands = candidates(platform, date)
    reports = load_reports(platform, date)
    st.title("Seller health assistant")
    st.caption(f"{platform.capitalize()}  |  as of {date}  |  {cands['sellers_scored']} sellers scored")
    view_assistant(platform, date, cands, reports)


main()
