"""The verifier: plain code, no AI, that decides which model sentences may be sent.

A sentence is kept only if ALL of these hold:

1. It lists at least one source.
2. Every source ID is one of the feed items for this event (so the model
   cannot cite something the system never read).
3. Every number in it appears in the cited items.
4. Every capitalised word in it (names, places, organisations) appears in the
   cited items.

Rejected sentences are returned with the reason, so they can be logged. They
are never sent.

What this does NOT prove: that the sentence is true or that the cited items
actually support its meaning. It proves that the specifics (who, where, how
many) came from the cited text and not from the model's imagination. Wording
rules in the prompt and the reader's access to the source links cover the rest.
The checks lean towards dropping a good sentence over keeping a bad one: for
example "2 million" will not match "2,000,000".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from news_brief.feeds import Item
from news_brief.summarise import Sentence

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20",
}
# Magnitude words must appear in the source as words ("2 million" needs "million").
MAGNITUDES = frozenset({"hundred", "thousand", "million", "billion", "trillion"})

# Capitalised words that carry no factual content when they start a sentence.
# A word missing from this list is checked against the source and, if absent,
# the sentence is dropped. That fails safe: the rejection log shows which
# harmless starters are being dropped so they can be added here.
FUNCTION_WORDS = frozenset("""
a an the this that these those in on at by for from with without after before
during while as but and or if when where which who what however meanwhile
also officials sources reports according it its he she they there here
following despite amid since until among throughout although because though
whereas instead about around nearly almost roughly over under more most some several many few
all both each another at least up to only neither either one
""".split())
# Deliberately NOT exempt: weekdays, months, "today", "yesterday" and other time
# words. A date or time is a factual claim and must come from the cited items.

# Fewer than this many shared leading letters is not treated as the same word.
MIN_SHARED_PREFIX = 5


@dataclass(frozen=True)
class Rejection:
    sentence: Sentence
    reason: str


def _normalise(text: str) -> str:
    """Make "U.S." and "U.K." single words ("US", "UK") instead of letters split at the dots."""
    return re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), text)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalise(text).lower())


def outlet_label(outlet: str) -> str:
    """"The Guardian (World)" -> "The Guardian": the name a sentence may attribute to."""
    return re.sub(r"\s*\(.*?\)\s*$", "", outlet)


def numbers_in(text: str) -> set[str]:
    """Numbers as digit strings (commas and % removed), number words converted."""
    found = set()
    for match in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        found.add(match.replace(",", "").rstrip("."))
    for word in _words(text):
        if word in NUMBER_WORDS:
            found.add(NUMBER_WORDS[word])
    return found


def magnitudes_in(text: str) -> set[str]:
    """Quantity words, with plurals folded in ("millions" counts as "million")."""
    found = set()
    for word in _words(text):
        singular = word[:-1] if word.endswith("s") else word
        if singular in MAGNITUDES:
            found.add(singular)
    return found


def names_in(text: str) -> list[str]:
    """Capitalised words, lower-cased, minus harmless function words."""
    names = []
    for word in re.findall(r"[A-Z][A-Za-z]*(?:-[A-Za-z]+)*", _normalise(text)):
        # "Arab-majority" is checked as "Arab"; only the capitalised parts are names.
        for part in word.split("-"):
            lowered = part.lower()
            if not part[0].isupper():
                continue
            if lowered in FUNCTION_WORDS or lowered in NUMBER_WORDS or lowered in MAGNITUDES:
                continue  # numbers and quantities are checked separately
            names.append(lowered)
    return names


def _in_source(word: str, source_words: set[str]) -> bool:
    """Exact match, or the same word with a different ending (Syria / Syrian)."""
    if word in source_words:
        return True
    if len(word) < MIN_SHARED_PREFIX:
        return False
    return any(
        len(other) >= MIN_SHARED_PREFIX
        and (other.startswith(word) or word.startswith(other))
        for other in source_words
    )


def verify_sentence(sentence: Sentence, allowed: dict[str, Item]) -> str | None:
    """Return the reason to reject the sentence, or None if it passes."""
    if not sentence.text.strip():
        return "empty sentence"
    if not sentence.sources:
        return "no sources listed"

    unknown = [s for s in sentence.sources if s not in allowed]
    if unknown:
        return f"source id not among this event's items: {', '.join(unknown)}"

    # The outlet's name counts as source text: "According to The Guardian" is
    # allowed when, and only when, the sentence cites a Guardian item.
    source_text = " ".join(
        f"{outlet_label(allowed[s].outlet)} {allowed[s].title} {allowed[s].summary}"
        for s in sentence.sources
    )

    missing_numbers = sorted(numbers_in(sentence.text) - numbers_in(source_text))
    if missing_numbers:
        return f"number not in cited items: {', '.join(missing_numbers)}"

    missing_magnitudes = sorted(magnitudes_in(sentence.text) - magnitudes_in(source_text))
    if missing_magnitudes:
        return f"quantity word not in cited items: {', '.join(missing_magnitudes)}"

    source_words = set(_words(source_text))
    missing_names = [n for n in names_in(sentence.text) if not _in_source(n, source_words)]
    if missing_names:
        return f"name not in cited items: {', '.join(dict.fromkeys(missing_names))}"

    return None


def verify(sentences: list[Sentence], allowed: dict[str, Item],
           ) -> tuple[list[Sentence], list[Rejection]]:
    """Split sentences into those that may be sent and those rejected."""
    accepted: list[Sentence] = []
    rejected: list[Rejection] = []
    for sentence in sentences:
        reason = verify_sentence(sentence, allowed)
        if reason is None:
            accepted.append(sentence)
        else:
            rejected.append(Rejection(sentence, reason))
    return accepted, rejected
