"""Deterministic intent classification.

No LLM, no network, no Anthropic dependency. Classification is exact phrase
matching over normalized text, grouped by intent so the rules stay readable
rather than becoming an unmaintainable pile of regexes.

Behaviour that matters:

* Exactly one intent family matched  -> that intent, gate may pass.
* Several materially different families matched -> `MULTIPLE_INTENTS`. The
  classifier does NOT pick a winner; guessing is exactly the failure mode the
  Claude fallback exists to handle.
* Nothing matched -> `NO_CLEAR_INTENT`.

`COMPLEX_SYNTHESIS` is the one legitimate multi-signal intent: an explicit
synthesis verb ("analyse ... considering ...") plus evidence from two or more
families is a single clear request, not a conflict.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from backend.app.agents.contracts import Intent, RationaleCode


def normalize_question(raw: str) -> str:
    """Casefold, strip accents, collapse punctuation and whitespace.

    Keeps `+`, `-` and digits: what-if questions like "+0.5" depend on them.
    """
    decomposed = unicodedata.normalize("NFKD", raw)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = without_accents.casefold()
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace() or ch in "+-.") else " " for ch in lowered)
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class IntentSignal:
    """One intent's trigger phrases, matched on word boundaries."""

    intent: Intent
    rationale: RationaleCode
    phrases: tuple[str, ...]

    def matches(self, normalized: str) -> bool:
        return any(_contains_phrase(normalized, phrase) for phrase in self.phrases)


def _contains_phrase(normalized: str, phrase: str) -> bool:
    """Word-boundary containment, so 'form' does not match 'information'."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized) is not None


# --------------------------------------------------------------------------
# Signal definitions, grouped per intent and kept deliberately readable.
# --------------------------------------------------------------------------
STANDINGS_SIGNAL = IntentSignal(
    Intent.SIMPLE_FACT,
    RationaleCode.MATCHED_STANDINGS,
    ("table", "standings", "league position", "top of the", "bottom of the", "who is top", "who's top"),
)

FIXTURES_SIGNAL = IntentSignal(
    Intent.SIMPLE_FACT,
    RationaleCode.MATCHED_FIXTURES,
    ("next fixture", "next match", "play next", "playing next", "fixtures", "when do", "when does",
     "latest result", "last result", "latest results", "who won"),
)

LIVE_SIGNAL = IntentSignal(
    Intent.LIVE_MATCH,
    RationaleCode.MATCHED_LIVE,
    ("right now", "at the moment", "live", "currently playing", "happening in this match",
     "current score", "what minute", "score now"),
)

HISTORICAL_SIGNAL = IntentSignal(
    Intent.HISTORICAL_STATISTICAL,
    RationaleCode.MATCHED_HISTORICAL,
    ("last five", "last 5", "recent form", "home form", "away form", "recent matches",
     "recent results", "compare", "head to head", "form guide"),
)

PREDICTION_SIGNAL = IntentSignal(
    Intent.PREDICTION,
    RationaleCode.MATCHED_PREDICTION,
    ("predict", "prediction", "who will win", "chances", "win probability", "likely result",
     "odds of winning", "expected result", "model confidence"),
)

EXPLANATION_SIGNAL = IntentSignal(
    Intent.PREDICTION_EXPLANATION,
    RationaleCode.MATCHED_EXPLANATION,
    ("why does the model", "why is the model", "explain the prediction", "explain why the model",
     "what drives the prediction", "model favour", "model favor", "why the model"),
)

SCORELINE_SIGNAL = IntentSignal(
    Intent.SCORELINE,
    RationaleCode.MATCHED_SCORELINE,
    ("most likely score", "likely scoreline", "scoreline", "expected goals", "correct score",
     "what score", "how many goals"),
)

RESEARCH_SIGNAL = IntentSignal(
    Intent.RESEARCH,
    RationaleCode.MATCHED_RESEARCH,
    ("injury", "injuries", "injured", "suspended", "suspension", "team news", "manager said",
     "press conference", "reports", "transfer", "doubtful", "fitness"),
)

TACTICAL_SIGNAL = IntentSignal(
    Intent.TACTICAL,
    RationaleCode.MATCHED_TACTICAL,
    ("tactically", "tactical", "press", "pressing", "buildup", "build up", "formation",
     "high line", "matchup", "match up", "style of play"),
)

WHAT_IF_SIGNAL = IntentSignal(
    Intent.WHAT_IF,
    RationaleCode.MATCHED_WHAT_IF,
    ("what if", "what would happen if", "how would the prediction change", "if instead",
     "suppose that", "hypothetically"),
)

SCOUTING_SIGNAL = IntentSignal(
    Intent.SCOUTING,
    RationaleCode.MATCHED_SCOUTING,
    ("similar to", "players like", "stylistically similar", "comparable players", "scout",
     "find players", "player similarity"),
)

# Ordered; order matters only for deterministic reporting, not precedence.
INTENT_SIGNALS: tuple[IntentSignal, ...] = (
    STANDINGS_SIGNAL,
    FIXTURES_SIGNAL,
    LIVE_SIGNAL,
    HISTORICAL_SIGNAL,
    PREDICTION_SIGNAL,
    EXPLANATION_SIGNAL,
    SCORELINE_SIGNAL,
    RESEARCH_SIGNAL,
    TACTICAL_SIGNAL,
    WHAT_IF_SIGNAL,
    SCOUTING_SIGNAL,
)

# An explicit request to combine evidence. Paired with >=2 evidence families
# this is a single coherent request, not a conflict.
SYNTHESIS_MARKERS: tuple[str, ...] = (
    "analyse", "analyze", "analysis", "considering", "taking into account",
    "overall assessment", "full picture", "combining",
)

# Asking *why* about a live match is reasoning, not plain state retrieval.
LIVE_REASONING_MARKERS: tuple[str, ...] = ("why", "struggling", "struggle", "better so far", "explain")

# Subsumption: `dominant -> intents it absorbs`. Two signals firing is only a
# CONFLICT when neither subsumes the other.
#
# The load-bearing rule is that SIMPLE_FACT is absorbed by every richer intent.
# A bare fact reference is how people SCOPE a richer question - "what injuries
# affect Arsenal's *next match*", "how will they set up in their *next match*" -
# so treating that reference as a rival intent would send ordinary research and
# tactical questions to the fallback classifier for no reason.
#
# PREDICTION absorbs SCORELINE because its plan already runs both the outcome
# and scoreline tools, so nothing is lost.
SUBSUMPTION: dict[Intent, tuple[Intent, ...]] = {
    Intent.PREDICTION_EXPLANATION: (Intent.PREDICTION, Intent.SCORELINE, Intent.SIMPLE_FACT),
    Intent.PREDICTION: (Intent.SCORELINE, Intent.SIMPLE_FACT),
    Intent.SCORELINE: (Intent.SIMPLE_FACT,),
    Intent.RESEARCH: (Intent.SIMPLE_FACT,),
    Intent.TACTICAL: (Intent.SIMPLE_FACT,),
    Intent.LIVE_MATCH: (Intent.SIMPLE_FACT,),
    Intent.HISTORICAL_STATISTICAL: (Intent.SIMPLE_FACT,),
    Intent.WHAT_IF: (Intent.SIMPLE_FACT, Intent.PREDICTION, Intent.SCORELINE),
    Intent.SCOUTING: (Intent.SIMPLE_FACT,),
}


def _apply_subsumption(matched: set[Intent]) -> set[Intent]:
    """Drop intents absorbed by a more specific one. Removals are computed from
    the ORIGINAL set so the result does not depend on iteration order."""
    absorbed: set[Intent] = set()
    for dominant in matched:
        absorbed.update(SUBSUMPTION.get(dominant, ()))
    return matched - absorbed


@dataclass(frozen=True)
class IntentClassification:
    """Deterministic classification outcome. `intent is None` means the router
    could not decide - never a guess."""

    intent: Intent | None
    matched_intents: tuple[Intent, ...]
    rationale_codes: tuple[RationaleCode, ...]
    failure_code: RationaleCode | None
    live_reasoning_requested: bool = False


def classify_intent(question: str) -> IntentClassification:
    """Classify one question deterministically."""
    normalized = normalize_question(question)

    matched: list[IntentSignal] = [signal for signal in INTENT_SIGNALS if signal.matches(normalized)]
    matched_intents = tuple(dict.fromkeys(signal.intent for signal in matched))
    rationale_codes = tuple(dict.fromkeys(signal.rationale for signal in matched))

    if not matched:
        return IntentClassification(
            intent=None,
            matched_intents=(),
            rationale_codes=(),
            failure_code=RationaleCode.NO_CLEAR_INTENT,
        )

    live_reasoning = bool(
        Intent.LIVE_MATCH in matched_intents
        and any(_contains_phrase(normalized, marker) for marker in LIVE_REASONING_MARKERS)
    )

    # Collapse subsumed intents before judging conflict, so a scoping fact
    # reference is never mistaken for a rival intent.
    effective = _apply_subsumption(set(matched_intents))

    # Legitimate multi-signal request: an explicit synthesis verb over >=2 families.
    has_synthesis_marker = any(_contains_phrase(normalized, m) for m in SYNTHESIS_MARKERS)
    if has_synthesis_marker and len(effective) >= 2:
        return IntentClassification(
            intent=Intent.COMPLEX_SYNTHESIS,
            matched_intents=matched_intents,
            rationale_codes=rationale_codes + (RationaleCode.MATCHED_COMPLEX_SYNTHESIS,),
            failure_code=None,
            live_reasoning_requested=live_reasoning,
        )

    if len(effective) > 1:
        # Materially different intents. Do NOT pick a winner.
        return IntentClassification(
            intent=None,
            matched_intents=matched_intents,
            rationale_codes=rationale_codes,
            failure_code=RationaleCode.MULTIPLE_INTENTS,
            live_reasoning_requested=live_reasoning,
        )

    intent = next(iter(effective))
    codes = rationale_codes
    if live_reasoning and intent is Intent.LIVE_MATCH:
        codes = codes + (RationaleCode.LIVE_REASONING_REQUESTED,)

    return IntentClassification(
        intent=intent,
        matched_intents=matched_intents,
        rationale_codes=codes,
        failure_code=None,
        live_reasoning_requested=live_reasoning,
    )


__all__ = [
    "INTENT_SIGNALS",
    "IntentClassification",
    "IntentSignal",
    "classify_intent",
    "normalize_question",
]
