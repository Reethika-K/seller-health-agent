"""
Seller Health Monitor - the web app.

    pip install -U streamlit
    streamlit run app.py          (run it from the project folder)

It only READS the reports that the pipeline wrote to reports/, plus the daily metrics, so opening
it costs nothing. Two places use Gemini, and only when you click: the "Run pipeline" button in the
sidebar, and the "Rewrite with Gemini" / Assistant features. Nothing is ever sent to a seller:
"Approve" just saves the wording in reports/approved_<platform>_<date>.json.

Views:  Priority list | Seller detail | Assistant | How it works
"""
import contextlib
import functools
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
os.chdir(BASE)                                   # the agents use relative paths (reports/, <platform>_data/)

st.set_page_config(page_title="Seller Health Monitor", layout="wide", initial_sidebar_state="expanded")

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

VIEWS = ["Priority list", "Seller detail", "Assistant", "How it works"]
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
    st.sidebar.caption("Opening the app never calls Gemini. Approving a message only saves it; nothing is sent to sellers.")
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


# ======================================================================== priority list
def build_rows(cands, reports, approvals, platform):
    rows = []
    for c in cands["candidates"]:
        sid = c["seller_id"]
        plan = reports["plans"].get(sid)
        confirmed = c["status"] == "confirmed"
        rows.append({
            "Priority": priority_of(c, plan),
            "Seller": sid,
            "Tier": c["size_tier"],
            "Status": "Confirmed" if confirmed else "Watch",
            "Main problem": c["primary_category"],
            "Also affected": ", ".join(c["secondary_categories"]) or "-",
            "Weeks flagged": c["primary_consecutive_weeks"],
            "Owner team": (plan["owner_team"] if plan else OWNER_TEAM[c["primary_category"]]) if confirmed else "-",
            "Diagnosed": "Yes" if sid in reports["diagnoses"] else "-",
            "Message": message_status(sid, plan, approvals) if confirmed else "-",
            "_overall": c["overall_score"],
        })
    order = {"P1": 0, "P2": 1, "Watch": 2}
    rows.sort(key=lambda r: (order[r["Priority"]], r["_overall"]))
    return pd.DataFrame(rows)


def goto_seller(sid):
    st.session_state["sel_seller"] = sid
    st.session_state["view"] = "Seller detail"


def view_priority(platform, date, cands, reports, approvals):
    rows = build_rows(cands, reports, approvals, platform)
    conf = rows[rows.Status == "Confirmed"]

    k = st.columns(5)
    k[0].metric("Confirmed declining", len(conf), help="Flagged in the same category for 2 or more weekly runs.")
    k[1].metric("On watch", int((rows.Status == "Watch").sum()), help="Flagged this week only; could be a one-off dip.")
    k[2].metric("Diagnosed", int((rows.Diagnosed == "Yes").sum()))
    k[3].metric("P1 (contact within 2 days)", int((conf.Priority == "P1").sum()))
    k[4].metric("Messages approved", int((rows.Message == "Approved").sum()))

    headline = reports["pipeline"].get("monitoring_summary")
    if headline:
        st.info(headline)
    elif not reports["plans"] and not reports["diagnoses"]:
        st.warning("No AI analysis is saved for this date yet. The list below comes from the scoring rules only. "
                   "Use **Run the pipeline** in the sidebar to diagnose the top sellers and draft messages.")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Confirmed sellers by main problem")
        counts = conf["Main problem"].value_counts().rename_axis("category").reset_index(name="sellers")
        if counts.empty:
            st.caption("No confirmed sellers on this date.")
        else:
            bars = alt.Chart(counts).mark_bar(color=BLUE).encode(
                y=alt.Y("category:N", sort="-x", title=None),
                x=alt.X("sellers:Q", title=None, axis=alt.Axis(tickMinStep=1)))
            labels = bars.mark_text(align="left", dx=3).encode(text="sellers:Q")
            st.altair_chart((bars + labels).properties(height=max(120, 34 * len(counts))), width="stretch")
    with right:
        st.subheader("Filters")
        f1, f2, f3 = st.columns(3)
        prio = f1.multiselect("Priority", ["P1", "P2", "Watch"], default=["P1", "P2"])
        cats = f2.multiselect("Main problem", CATEGORIES)
        only_diag = f3.toggle("Only diagnosed sellers", value=False)
        st.caption("Tick 'Watch' in Priority to include sellers that were flagged for only one week.")

    view = rows[rows.Priority.isin(prio)]
    if cats:
        view = view[view["Main problem"].isin(cats)]
    if only_diag:
        view = view[view.Diagnosed == "Yes"]
    st.subheader(f"Sellers ({len(view)})")
    shown = view.drop(columns=["_overall"]).reset_index(drop=True)
    event = st.dataframe(shown, width="stretch", hide_index=True, on_select="rerun",
                         selection_mode="single-row", key=f"table_{platform}_{date}",
                         height=min(40 + 35 * max(len(shown), 1), 520))
    picked = event.selection.rows if event and event.selection else []
    if picked:
        sid = shown.loc[picked[0], "Seller"]
        st.button(f"Open {sid} in Seller detail", type="primary", on_click=goto_seller, args=(sid,))
    else:
        st.caption("Click a row to select a seller, then open the detail page.")


# ======================================================================== seller detail
def metric_chart(daily, metric, as_of, onset):
    s = daily[daily.date <= as_of][["date", metric]].copy()
    s["avg7"] = s[metric].rolling(7, min_periods=3).mean()
    base = s[metric].iloc[:21].mean()
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", tickCount=5))
    y = alt.Y(f"{metric}:Q", title=None, scale=alt.Scale(zero=False))
    layers = [
        alt.Chart(s).mark_line(color=GREY, opacity=0.5, strokeWidth=1).encode(x=x, y=y),
        alt.Chart(s).mark_line(color=BLUE, strokeWidth=2.5).encode(x=x, y=alt.Y("avg7:Q", title=None)),
        alt.Chart(pd.DataFrame({"base": [base]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(y="base:Q"),
    ]
    if onset:
        layers.append(alt.Chart(pd.DataFrame({"date": [pd.Timestamp(onset)]}))
                      .mark_rule(color=RED, strokeDash=[6, 3], strokeWidth=2).encode(x="date:T"))
    better = "higher is better" if metric in HIGHER_IS_BETTER else "lower is better"
    return alt.layer(*layers).properties(height=190, title=alt.TitleParams(pretty(metric), subtitle=better,
                                                                           anchor="start", fontSize=14))


def score_history_chart(hist, highlight):
    rows = []
    for h in hist:
        for c in CATEGORIES:
            if h[c] is not None:
                rows.append({"run": pd.Timestamp(h["as_of"]), "category": c, "score": h[c],
                             "group": c if c in highlight else "Other categories"})
    d = pd.DataFrame(rows)
    palette = [BLUE, AMBER, "#7c3aed", "#059669"]
    domain = list(highlight) + ["Other categories"]
    rng = palette[:len(highlight)] + ["#d1d5db"]
    line = alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("run:T", title="Weekly run", axis=alt.Axis(format="%d %b", tickCount=6)),
        y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1.02])),
        color=alt.Color("group:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None, orient="bottom")),
        detail="category:N",
        opacity=alt.condition(alt.datum.group == "Other categories", alt.value(0.7), alt.value(1.0)))
    rule = alt.Chart(pd.DataFrame({"y": [FLAG_SCORE]})).mark_rule(color=RED, strokeDash=[5, 3]).encode(y="y:Q")
    note = alt.Chart(pd.DataFrame({"y": [FLAG_SCORE], "t": ["flag line (0.35)"]})).mark_text(
        align="left", dy=-6, color=RED, x=4).encode(y="y:Q", text="t:N")
    return (line + rule + note).properties(height=260)


def view_detail(platform, date, cands, reports, approvals):
    by_id = {c["seller_id"]: c for c in cands["candidates"]}
    options = [c["seller_id"] for c in cands["candidates"] if c["status"] == "confirmed"] + \
              [c["seller_id"] for c in cands["candidates"] if c["status"] != "confirmed"]
    if not options:
        st.info("No flagged sellers on this date.")
        return
    if st.session_state.get("sel_seller") not in options:
        diagnosed = [s for s in options if s in reports["diagnoses"]]
        st.session_state["sel_seller"] = (diagnosed or options)[0]

    def label(sid):
        c = by_id[sid]
        return f"{sid}  -  {priority_of(c, reports['plans'].get(sid))}  -  {c['primary_category']}"

    sid = st.selectbox("Seller", options, format_func=label, key="sel_seller")
    c, diag, plan = by_id[sid], reports["diagnoses"].get(sid), reports["plans"].get(sid)
    alert = reports["alerts"].get(sid)
    use_platform(platform)
    sellers = sellers_table(platform)
    as_of = pd.Timestamp(date)
    prio = priority_of(c, plan)

    # ------------------------------------------------------------ header
    colour = {"P1": "red", "P2": "orange", "Watch": "gray"}[prio]
    st.markdown(f"## {sid}  :{colour}-badge[{prio}]  :blue-badge[{c['status'].capitalize()}]")
    st.caption(f"{sellers.loc[sid, 'category']}  |  {c['size_tier']} seller  |  joined {sellers.loc[sid, 'join_date']}")

    facts = dt.category_facts(sid, c["primary_category"], as_of) if dt else {}
    onset = (plan or diag or {}).get("estimated_onset_date") or facts.get("estimated_onset_date")
    speed = (plan or diag or {}).get("decline_speed") or facts.get("decline_speed", "unclear")
    team = plan["owner_team"] if plan else (OWNER_TEAM[c["primary_category"]] if c["status"] == "confirmed" else "-")
    facts_row = [("Main problem", c["primary_category"]), ("Also affected", ", ".join(c["secondary_categories"]) or "None"),
                 ("Started about", onset or "unclear"), ("Speed", speed.capitalize()), ("Owner team", team)]
    for col, (name, value) in zip(st.columns(5), facts_row):
        col.caption(name)
        col.markdown(f"**{value}**")

    if c["status"] != "confirmed":
        st.info("This seller is on watch: one category was flagged this week only. It is not diagnosed because it may "
                "be a short dip. If it is flagged again next week it becomes a confirmed case.")

    # ------------------------------------------------------------ analysis
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("What is going on")
        if diag:
            st.write(diag["summary"])
            titles = kb_titles()
            for rc in diag["root_causes"]:
                with st.expander(f"{rc['cause']}   (confidence: {rc['confidence']})", expanded=rc is diag["root_causes"][0]):
                    for ev in rc["evidence"]:
                        st.markdown(f"- {ev}")
                    if rc["knowledge_docs"]:
                        st.caption("Based on knowledge documents: " + ", ".join(
                            f"{d} ({titles[d]['title']})" if d in titles else d for d in rc["knowledge_docs"]))
            if diag.get("secondary_assessment"):
                st.markdown("**Secondary category**")
                st.write(diag["secondary_assessment"])
            if diag.get("weak_secondary_signals"):
                st.caption("Weak signals not treated as problems: " + ", ".join(diag["weak_secondary_signals"]))
        else:
            if alert:
                st.write(alert["reasoning"])
                if alert.get("evidence"):
                    st.markdown("**Evidence from the monitoring step**")
                    for ev in alert["evidence"]:
                        st.markdown(f"- {ev}")
                if not alert.get("investigated", True):
                    st.caption("The monitoring agent did not look at this seller in detail; this text comes from the scoring rules.")
            else:
                st.write("No AI analysis is saved for this seller.")
            if c["status"] == "confirmed":
                st.caption("This confirmed seller has not been diagnosed yet. Raise 'Sellers to diagnose per run' in the "
                           "sidebar and run the pipeline to include them.")
            if facts.get("elevated_metrics"):
                st.markdown("**Metrics clearly worse than this seller's own first three weeks**")
                st.dataframe(pd.DataFrame([{"Metric": pretty(m["metric"]), "Started": m["onset_date"],
                                            "Strength": m["strength"]} for m in facts["elevated_metrics"]]),
                             hide_index=True, width="stretch")
    with right:
        st.subheader("What to do")
        if plan:
            st.write(plan["internal_summary"])
            st.table(pd.DataFrame([{"Action": a["action"], "Owner": a["owner"].replace("_", " "),
                                    "Due in (days)": a["due_in_days"]} for a in plan["actions"]]).style.hide(axis="index"))
            st.markdown(f"**Contact the seller within {plan['first_contact_within_days']} days.**  Priority {plan['priority']}, "
                        f"owner: {plan['owner_team']}.")
            st.markdown("**Check again next week:** " + ", ".join(pretty(m) for m in plan["metrics_to_recheck"]))
            st.markdown("**Escalate if:** " + plan["escalate_if"])
        elif c["status"] == "confirmed":
            days = PRIORITY_RULE[c["severity"]][1]
            st.write(f"No action plan yet. By the rules this is **{prio}**: contact the seller within {days} days. "
                     f"Owner team: {OWNER_TEAM[c['primary_category']]}.")
        else:
            st.write("No action is needed yet. Re-check next week.")

    # ------------------------------------------------------------ charts
    st.subheader("The numbers")
    cats_here = [c["primary_category"]] + [x for x in c["secondary_categories"]]
    other = [x for x in CATEGORIES if x not in cats_here]
    cat = st.radio("Category", cats_here + other, horizontal=True, key=f"cat_{platform}_{date}_{sid}",
                   format_func=lambda x: x + (" (main)" if x == c["primary_category"] else ""))
    daily = mt._load()["df"]
    daily = daily[daily.seller_id == sid].sort_values("date")
    cat_onset = onset if cat == c["primary_category"] else (
        dt.category_facts(sid, cat, as_of).get("estimated_onset_date") if dt else None)
    cols = st.columns(len(METRICS[cat]))
    for col, m in zip(cols, METRICS[cat]):
        col.altair_chart(metric_chart(daily, m, as_of, cat_onset), width="stretch")
    st.caption("Light grey: daily value. Blue: 7-day average. Grey dashed: average of the seller's first 21 days. "
               "Red dashed: estimated start of the decline." + ("" if cat_onset else " (No clear start for this category.)"))

    hist = mt.get_seller_history(sid, date, 6)
    if "history" in hist:
        st.markdown("**Category scores over the last weekly runs** (lower is worse; at or below the red line a category is flagged)")
        st.altair_chart(score_history_chart(hist["history"][::-1], cats_here[:4]), width="stretch")

    # ------------------------------------------------------------ message
    if plan:
        message_block(platform, date, sid, plan, diag, approvals)


def message_block(platform, date, sid, plan, diag, approvals):
    st.subheader("Message to the seller")
    k = f"{platform}_{date}_{sid}"
    ks, kb_ = f"subj_{k}", f"body_{k}"
    saved = approvals.get(sid)
    original = plan["seller_message"]
    if ks not in st.session_state:
        st.session_state[ks] = (saved or original)["subject"]
        st.session_state[kb_] = (saved or {"body": original["body"]})["body"]

    def reset():
        st.session_state[ks], st.session_state[kb_] = original["subject"], original["body"]
        data = load_approvals(platform, date)
        data.pop(sid, None)
        save_approvals(platform, date, data)

    def approve():
        data = load_approvals(platform, date)
        data[sid] = {"subject": st.session_state[ks], "body": st.session_state[kb_], "approved": True,
                     "approved_at": datetime.now().isoformat(timespec="seconds")}
        save_approvals(platform, date, data)

    def rewrite():
        instruction = st.session_state.get(f"instr_{k}", "").strip() or "Make it clearer and a little shorter."
        try:
            subj, body = rewrite_message(st.session_state[ks], st.session_state[kb_], instruction, plan)
            st.session_state[ks], st.session_state[kb_] = subj, body
            st.session_state.pop(f"rw_err_{k}", None)
        except Exception as exc:
            st.session_state[f"rw_err_{k}"] = str(exc)[:300]

    st.text_input("Subject", key=ks)
    st.text_area("Message", key=kb_, height=230)
    words = len(st.session_state[kb_].split())
    problems = message_problems(st.session_state[ks], st.session_state[kb_])
    st.caption(f"{words} words (limit 150).")
    for p in problems:
        st.warning(p)

    current = (st.session_state[ks], st.session_state[kb_])
    if saved and saved.get("approved"):
        if current == (saved["subject"], saved["body"]):
            st.success(f"Approved on {saved['approved_at']}. Saved to reports/approved_{platform}_{date}.json. "
                       "It has not been sent: this app does not send messages.")
        else:
            st.info("You changed the text after approving. Approve again to save the new wording.")

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Approve message", type="primary", on_click=approve, disabled=bool(problems))
    b2.button("Reset to original", on_click=reset)

    with st.expander("Rewrite with Gemini"):
        st.text_input("What should change?", key=f"instr_{k}", placeholder="for example: friendlier, shorter, more specific about the first step")
        st.button("Rewrite", on_click=rewrite, disabled=bool(AGENT_ERROR))
        if st.session_state.get(f"rw_err_{k}"):
            st.error("Could not rewrite: " + st.session_state[f"rw_err_{k}"])
        st.caption("The new wording is checked against the same rules as the original (no internal terms, under 150 words) "
                   "and replaces the text above. You still have to approve it.")


def message_problems(subject, body):
    problems = []
    low = (subject + " " + body).lower()
    for w in JARGON:
        if w in low:
            problems.append(f"The message contains the internal term '{w}'. Remove it before approving.")
    if dg and dg.DOC_ID.search(subject + " " + body):
        problems.append("The message contains a document ID (like MKT-01). Remove it.")
    if len(body.split()) > 150:
        problems.append("The message is longer than 150 words.")
    if not subject.strip() or not body.strip():
        problems.append("Subject and message cannot be empty.")
    return problems


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


def rewrite_message(subject, body, instruction, plan):
    from google.genai import types
    system = ("You rewrite a message from a marketplace account manager to a seller whose performance has dropped. "
              "Keep it polite, supportive and specific. Keep the facts the same and do not invent causes, numbers or promises. "
              "Under 150 words. NEVER use these words: flagged, sub-score, AHP, diagnostic, knowledge base, ground truth. "
              "Do not include internal scores, tool names or document IDs.")
    config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                         response_schema=dg.SellerMessage, temperature=0.4)
    feedback = ""
    for _ in range(3):
        prompt = (f"Problem area: {plan['primary_category']}. Instruction: {instruction}\n\n"
                  f"CURRENT SUBJECT: {subject}\nCURRENT MESSAGE:\n{body}" +
                  (f"\n\nYour previous attempt was rejected: {feedback}. Fix this." if feedback else ""))
        resp = gemini_call(lambda: dg.client.models.generate_content(model=MODEL, contents=prompt, config=config))
        try:
            m = dg.SellerMessage.model_validate_json(resp.text)
        except Exception as exc:
            feedback = str(exc)[:200]
            continue
        problems = message_problems(m.subject, m.body)
        if not problems and len(m.subject) < 90:
            return m.subject, m.body
        feedback = "; ".join(problems) or "subject must be under 90 characters"
    raise RuntimeError("Gemini did not produce a valid rewrite after 3 tries: " + feedback)


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
  it in the app (Run the pipeline in the sidebar; Approve in Seller detail).
- If asked to rewrite a seller message, write it in your reply: polite, specific, under 150 words, and never mention scores, internal
  tools, document IDs or the words flagged, sub-score, AHP, diagnostic, knowledge base. Say the user can paste it into the message box.
- Keep answers short and in plain language. Mention the seller ID and the date when you give facts."""


def new_chat(platform, date):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=chat_system(platform, date), tools=make_chat_tools(platform, date), temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=8))
    return dg.client.chats.create(model=MODEL, config=cfg)


def view_assistant(platform, date, cands, reports):
    st.subheader("Ask about your sellers")
    st.caption("Answers come from the same data and reports as the other pages, through read-only tools. "
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


# ======================================================================== how it works
PIPELINE_DOT = """
digraph G {
  rankdir=TB; nodesep=0.25; ranksep=0.35; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor="#111827", color="#9ca3af"];
  edge [color="#6b7280"];
  data  [label="Daily metrics\\n15 per seller", fillcolor="#f3f4f6"];
  score [label="Scoring (code)\\nAHP weights x levels", fillcolor="#dbeafe"];
  mon   [label="Monitor\\ncode flags, LLM explains", fillcolor="#fef3c7"];
  sel   [label="Select (code)\\ntop N confirmed", fillcolor="#dbeafe"];
  diag  [label="Diagnose\\ncode timeline + LLM + knowledge base", fillcolor="#fef3c7"];
  draft [label="Draft\\ncode rules + LLM wording", fillcolor="#fef3c7"];
  human [label="You review\\nedit, approve", fillcolor="#dcfce7"];
  data -> score -> mon -> sel -> diag -> draft -> human;
}
"""

CODE_VS_LLM = pd.DataFrame([
    ["Scoring and flagging", "Scores every seller, flags categories at or below 0.35, decides confirmed vs watch", "Nothing"],
    ["Main problem and severity", "Longest flagged run wins; Critical if the score is below 0.15", "Nothing"],
    ["Start date and speed", "Measures when each metric left the seller's own baseline and how fast", "Nothing"],
    ["Evidence and causes", "Supplies the numbers and the timeline", "Investigates with tools, searches the knowledge base, explains the likely causes"],
    ["Priority, deadline, owner", "Fixed rules: Critical is P1 (2 days), At-risk is P2 (7 days); owner by category", "Nothing"],
    ["Wording", "Checks length, forbidden words, valid metric names; retries on a bad draft", "Writes the summary, actions and seller message"],
    ["Sending", "Nothing is sent. You approve each message", "Nothing"],
], columns=["Step", "Code decides", "LLM does"])


def view_how(platform, date, cands, reports):
    st.subheader("How the pipeline works")
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.graphviz_chart(PIPELINE_DOT, width="stretch")
        st.caption("Blue: plain code. Amber: code plus the LLM. Green: a person.")
    with right:
        st.markdown("Small language models are unreliable at applying rules across many records, so code applies the rules "
                    "and does the arithmetic. The LLM investigates one seller at a time, uses the tools, and explains in words.")
        st.table(CODE_VS_LLM.style.hide(axis="index"))

    failures = reports["pipeline"].get("failures") or []
    if failures:
        st.subheader("Problems in the last run")
        for f in failures:
            st.warning(f)
    if st.session_state.get("run_log"):
        with st.expander("Log of the last run from this app"):
            st.code(st.session_state["run_log"][-6000:])

    with st.expander("Evaluation against the hidden ground truth (testing only)"):
        evaluation(platform, date, cands, reports)


def evaluation(platform, date, cands, reports):
    st.caption("Possible only because the data is synthetic: we know which sellers were really made to decline. "
               "A real system would not have this file.")
    path = BASE / f"{platform}_data" / "ground_truth.csv"
    if not path.exists():
        st.write("No ground_truth.csv for this platform.")
        return
    gt = pd.read_csv(path, parse_dates=["decline_start_date"]).set_index("seller_id")
    as_of = pd.Timestamp(date)
    conf = [c for c in cands["candidates"] if c["status"] == "confirmed"]
    real = [c for c in conf if c["seller_id"] in gt.index and gt.loc[c["seller_id"], "is_at_risk"] == 1]
    started = gt[(gt.is_at_risk == 1) & (gt.decline_start_date <= as_of)]
    rows = [("Confirmed sellers that are truly declining", f"{len(real)} of {len(conf)} ({100 * len(real) / max(len(conf), 1):.0f}%)"),
            ("Truly declining sellers (started by this date) that were confirmed",
             f"{len({c['seller_id'] for c in real} & set(started.index))} of {len(started)}")]
    if real:
        strict = sum(c["primary_category"] == gt.loc[c["seller_id"], "primary_category"] for c in real)
        lenient = sum(c["primary_category"] in (gt.loc[c["seller_id"], "primary_category"],
                                                gt.loc[c["seller_id"], "secondary_category"]) for c in real)
        rows.append(("Main problem correct (strict: equals the true main problem)", f"{strict} of {len(real)}"))
        rows.append(("Main problem correct (lenient: also accepts the true secondary)", f"{lenient} of {len(real)}"))
    errs, speed_ok, n = [], 0, 0
    for sid, d in reports["diagnoses"].items():
        if sid in gt.index and gt.loc[sid, "is_at_risk"] == 1 and d.get("estimated_onset_date"):
            errs.append((pd.Timestamp(d["estimated_onset_date"]) - gt.loc[sid, "decline_start_date"]).days)
            n += 1
            speed_ok += d["decline_speed"] == gt.loc[sid, "decline_speed"]
    if errs:
        rows.append(("Estimated start date, median error (positive = estimated too late)", f"{pd.Series(errs).median():+.0f} days"))
        rows.append(("Start date within 7 days of the truth", f"{sum(abs(e) <= 7 for e in errs)} of {len(errs)}"))
        rows.append(("Speed label (sudden or gradual) correct", f"{speed_ok} of {n}"))
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Result"]), hide_index=True, width="stretch")
    st.caption("Confirmed sellers that are not truly declining are false alarms. Sellers whose decline is still very recent cannot be confirmed yet, because a category must be flagged in 2 weekly runs.")


# ======================================================================== main
def main():
    platform, date, _top_n = sidebar()
    cands = candidates(platform, date)
    reports = load_reports(platform, date)
    approvals = load_approvals(platform, date)

    st.title("Seller health")
    st.caption(f"{platform.capitalize()}  |  as of {date}  |  {cands['sellers_scored']} sellers scored")

    if st.session_state.get("view") not in VIEWS:
        st.session_state["view"] = VIEWS[0]
    view = st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")
    st.divider()
    if view == "Priority list":
        view_priority(platform, date, cands, reports, approvals)
    elif view == "Seller detail":
        view_detail(platform, date, cands, reports, approvals)
    elif view == "Assistant":
        view_assistant(platform, date, cands, reports)
    else:
        view_how(platform, date, cands, reports)


main()
