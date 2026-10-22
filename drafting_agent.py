"""
Drafting agent: turns each diagnosis into an action plan and a message to the seller.
It uses STRUCTURED OUTPUT: the LLM must return JSON that matches a Pydantic schema, and the
result is checked by validators before it is accepted.

  python drafting_agent.py 2026-09-08                 draft for every seller in the diagnosis report
  python drafting_agent.py 2026-09-08 --seller S0065  draft for one seller

Input : reports/diagnosis_<platform>_<date>.json   (written by diagnostic_agent.py)
Output: reports/actions_<platform>_<date>.json

How the work is split:
  CODE   decides priority, due date and owner team from the severity and category (fixed rules),
         and picks the knowledge documents that ground the actions.
  LLM    writes the wording: summary, actions, the seller message, what to re-check.
  CHECK  validators reject a draft that breaks the rules (too long, jargon, unknown metric names,
         bad due dates). The agent then retries with the error as feedback.
"""
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError, model_validator

import kb
import monitor_tools as mt
from ahp_config import METRICS
from levels import ALL_METRICS

load_dotenv()
MODEL = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 3


def make_client():
    opts = {"timeout": 90_000}
    if hasattr(types, "HttpRetryOptions"):
        opts["retry_options"] = types.HttpRetryOptions(attempts=1)
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "missing"),
                        http_options=types.HttpOptions(**opts))


client = make_client()

# ---------------------------------------------------------------- rules decided by code
OWNER_TEAM = {"Marketing": "Catalog and ads team", "Logistics": "Logistics operations",
              "Customer Experience": "Customer experience team", "Seller support": "Seller support team",
              "Finance": "Finance operations", "Inventory": "Seller supply team"}
PRIORITY = {"Critical": ("P1", 2), "At-risk": ("P2", 7)}    # severity -> (priority, days to first contact)
JARGON = ["sub-score", "subscore", "ahp", "knowledge base", "diagnostic", "ground truth", "flagged"]
DOC_ID = re.compile(r"\b[A-Z]{3}-\d{2}\b")


# ---------------------------------------------------------------- the structured output
class Action(BaseModel):
    action: str = Field(description="one concrete step, starting with a verb")
    owner: Literal["seller", "platform_team"]
    due_in_days: int = Field(description="days from today, between 1 and 14")


class SellerMessage(BaseModel):
    subject: str = Field(description="under 90 characters")
    body: str = Field(description="under 150 words, polite and specific")


class DraftNarrative(BaseModel):
    internal_summary: str = Field(description="2-3 sentences for the account manager, under 90 words")
    actions: List[Action] = Field(description="1 to 3 actions, most urgent first")
    seller_message: SellerMessage
    metrics_to_recheck: List[str] = Field(description="metric names to look at again next week")
    escalate_if: str = Field(description="one sentence: what would mean this needs escalation")

    @model_validator(mode="after")
    def check_rules(self):
        problems = []
        if not 1 <= len(self.actions) <= 3:
            problems.append("give between 1 and 3 actions")
        for a in self.actions:
            if not 1 <= a.due_in_days <= 14:
                problems.append(f"due_in_days must be between 1 and 14 (got {a.due_in_days})")
        if len(self.seller_message.body.split()) > 150:
            problems.append("seller_message.body must be 150 words or fewer")
        if len(self.seller_message.subject) > 90:
            problems.append("seller_message.subject must be under 90 characters")
        if len(self.internal_summary.split()) > 90:
            problems.append("internal_summary must be 90 words or fewer")
        text = (self.seller_message.subject + " " + self.seller_message.body).lower()
        for word in JARGON:
            if word in text:
                problems.append(f"the seller message must not contain the internal term '{word}'")
        if DOC_ID.search(self.seller_message.subject + " " + self.seller_message.body):
            problems.append("the seller message must not contain document IDs like MKT-01")
        bad = [m for m in self.metrics_to_recheck if m not in ALL_METRICS]
        if bad or not self.metrics_to_recheck:
            problems.append(f"metrics_to_recheck must be a non-empty list of valid metric names; "
                            f"invalid: {bad}. Valid names: {ALL_METRICS}")
        if problems:
            raise ValueError("; ".join(problems))
        return self


class ActionPlan(BaseModel):
    seller_id: str
    as_of_date: str
    primary_category: str
    secondary_categories: List[str]
    estimated_onset_date: str | None
    decline_speed: str
    # decided by code
    priority: Literal["P1", "P2"]
    owner_team: str
    first_contact_within_days: int
    # written by the LLM and validated
    internal_summary: str
    actions: List[Action]
    seller_message: SellerMessage
    metrics_to_recheck: List[str]
    escalate_if: str
    attempts_needed: int


SYSTEM = """You are the Drafting Agent in a seller-health system for an e-commerce marketplace.
You receive a diagnosis of a declining seller and write (1) an internal action plan for the platform's
account manager and (2) a short message to the seller. Return JSON that matches the schema.

RULES
- Base the actions on the diagnosis and on the "first actions" in the knowledge documents you are given.
  Do not invent causes or numbers that are not in the diagnosis.
- The seller message is polite and supportive and not accusatory. Say in plain words what we noticed
  (for example "your product listing quality has dropped"), give the first one or two things to do, and
  offer help. Do not mention internal scores, internal tools, the word "flagged", or document IDs.
- If the diagnosis is not certain, say "may" or "appears" in the message instead of stating a cause as fact.
- Write 1 to 3 actions: concrete steps starting with a verb. Each has an owner (seller or platform_team)
  and a due date in days between 1 and 14, most urgent first. Do NOT include the first contact with the
  seller or the message itself: that step is added automatically.
- In the seller message, NEVER use these words: flagged, sub-score, AHP, diagnostic, knowledge base,
  ground truth. Say "we noticed" or "we have seen" instead of "flagged".
- metrics_to_recheck uses only metric names from the allowed list you are given."""


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


def rules_for(cand):
    priority, days = PRIORITY[cand["severity"]]
    return {"priority": priority, "first_contact_within_days": days,
            "owner_team": OWNER_TEAM[cand["primary_category"]]}


def grounding_docs(diag):
    """The knowledge documents the diagnosis cited (by ID), so the actions are based on them."""
    docs = {d["id"]: d["text"] for d in kb.load_documents()}
    ids = []
    for rc in diag["root_causes"]:
        ids += [i for i in rc["knowledge_docs"] if i in docs and i not in ids]
    return [docs[i] for i in ids[:3]]


def draft(diag, cand, as_of):
    rules = rules_for(cand)
    task = (f"Seller {diag['seller_id']} on {mt.PLATFORM}, as of {as_of}.\n"
            f"Primary category: {diag['primary_category']}. Clear secondary: {diag['secondary_categories'] or 'none'}. "
            f"Weak signals, do not mention as problems: {diag.get('weak_secondary_signals') or 'none'}.\n"
            f"Decline started about {diag['estimated_onset_date']} and is {diag['decline_speed']}.\n"
            f"Secondary is a knock-on effect: {diag['secondary_is_knock_on']}.\n"
            f"Priority {rules['priority']}. Owner team: {rules['owner_team']}.\n\n"
            f"DIAGNOSIS SUMMARY: {diag['summary']}\n"
            f"LIKELY CAUSES: {json.dumps(diag['root_causes'])}\n"
            f"SECONDARY ASSESSMENT: {diag['secondary_assessment']}\n\n"
            f"KNOWLEDGE DOCUMENTS (use their first actions):\n" + "\n---\n".join(grounding_docs(diag)) +
            f"\n\nALLOWED METRIC NAMES: {ALL_METRICS}")
    config = types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                         response_schema=DraftNarrative, temperature=0.2)
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        contents = task + (f"\n\nYour previous draft was rejected: {feedback}\nFix these problems." if feedback else "")
        resp = _generate(contents=contents, config=config)
        try:
            n = DraftNarrative.model_validate_json(resp.text)
            # secondary categories are re-checked too: add their two most important metrics (a rule, not wording)
            for cat in diag["secondary_categories"]:
                n.metrics_to_recheck += [m for m in METRICS[cat][:2] if m not in n.metrics_to_recheck]
            contact = Action(action=f"Contact the seller about the {diag['primary_category'].lower()} problem "
                                    f"and offer support", owner="platform_team",
                             due_in_days=rules["first_contact_within_days"])
            n.actions = sorted([contact] + n.actions, key=lambda a: a.due_in_days)   # contact step added by code
            return ActionPlan(seller_id=diag["seller_id"], as_of_date=as_of,
                              primary_category=diag["primary_category"],
                              secondary_categories=diag["secondary_categories"],
                              estimated_onset_date=diag["estimated_onset_date"],
                              decline_speed=diag["decline_speed"], **rules,
                              **n.model_dump(), attempts_needed=attempt)
        except (ValidationError, ValueError) as e:
            feedback = str(e)[:600]
            print(f"   draft rejected (attempt {attempt}): {feedback[:150]}", flush=True)
    raise RuntimeError(f"No valid draft for {diag['seller_id']} after {MAX_ATTEMPTS} attempts")


def show(p):
    print(f"\n{p.seller_id}  {p.priority}  {p.owner_team}  (contact within {p.first_contact_within_days} days)")
    print(f"  {p.internal_summary}")
    for a in p.actions:
        print(f"  [{a.owner}, {a.due_in_days}d] {a.action}")
    print(f"  Re-check: {', '.join(p.metrics_to_recheck)}")
    print(f"  Escalate if: {p.escalate_if}")
    print(f"  --- message to seller ---\n  Subject: {p.seller_message.subject}\n  {p.seller_message.body}")


if __name__ == "__main__":
    as_of = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "2026-09-08"
    if "--platform" in sys.argv:
        mt.set_platform(sys.argv[sys.argv.index("--platform") + 1])
    path = Path("reports") / f"diagnosis_{mt.PLATFORM}_{as_of}.json"
    if not path.exists():
        sys.exit(f"{path} not found. Run diagnostic_agent.py {as_of} first.")
    diagnoses = json.loads(path.read_text())
    if "--seller" in sys.argv:
        diagnoses = [d for d in diagnoses if d["seller_id"] == sys.argv[sys.argv.index("--seller") + 1]]
    cands = {c["seller_id"]: c for c in mt.get_flagged_candidates(as_of)["candidates"]}
    plans = []
    for diag in diagnoses:
        print(f"Drafting for {diag['seller_id']}...", flush=True)
        plan = draft(diag, cands[diag["seller_id"]], as_of)
        plans.append(plan)
        show(plan)
        time.sleep(3)
    out = Path("reports") / f"actions_{mt.PLATFORM}_{as_of}.json"
    out.write_text(json.dumps([p.model_dump() for p in plans], indent=2))
    first_try = sum(p.attempts_needed == 1 for p in plans)
    print(f"\nSaved {out}\nValid on the first attempt: {first_try}/{len(plans)}")