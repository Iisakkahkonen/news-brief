"""Group feed items that report the same event.

Plain TF-IDF cosine similarity over headline and summary text, standard
library only. Each item joins the existing group whose combined word profile
it matches best, or starts a new group when nothing is similar enough.

The threshold is a tuning knob. It has to be set against real feeds, so the
command line can print the groups it makes (python -m news_brief --show-groups).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from news_brief.feeds import Item

DEFAULT_THRESHOLD = 0.30

STOPWORDS = frozenset("""
a about after again against all also an and any are as at be been before being
between both but by can could did do does during each for from further had has
have he her here him his how i if in into is it its just more most new news no
nor not now of off on once only or other our out over own said say says same
she should so some such than that the their them then there these they this
those through to too under until up us very was we were what when where which
while who whom why will with would you your
""".split())


def _stem(word: str) -> str:
    """Very light plural stripping so 'talks' and 'talk' match."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(w) for w in words if len(w) >= 3 and w not in STOPWORDS]


@dataclass
class Cluster:
    items: list[Item] = field(default_factory=list)

    @property
    def outlets(self) -> set[str]:
        return {item.outlet for item in self.items}

    @property
    def lead(self) -> Item:
        """The item shown as the event's headline: the earliest published."""
        dated = [i for i in self.items if i.published is not None]
        return min(dated, key=lambda i: i.published) if dated else self.items[0]

    def by_id(self) -> dict[str, Item]:
        return {item.item_id: item for item in self.items}


def _term_counts(item: Item) -> Counter:
    counts = Counter(tokens(item.title))
    counts.update(counts)  # headline words count double
    counts.update(tokens(item.summary))
    return counts


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vector.values()))


def group_items(items: list[Item], threshold: float = DEFAULT_THRESHOLD) -> list[Cluster]:
    """Cluster items by similarity. Biggest events (most outlets) come first."""
    if not items:
        return []

    counts = [_term_counts(item) for item in items]
    doc_freq: Counter = Counter()
    for c in counts:
        doc_freq.update(c.keys())
    n = len(items)
    idf = {term: math.log((n + 1) / (df + 1)) + 1 for term, df in doc_freq.items()}

    vectors = [{t: tf * idf[t] for t, tf in c.items()} for c in counts]

    clusters: list[Cluster] = []
    sums: list[dict[str, float]] = []
    for item, vec in zip(items, vectors):
        best, best_score = -1, 0.0
        for index, total in enumerate(sums):
            score = _cosine(vec, total)
            if score > best_score:
                best, best_score = index, score
        if best >= 0 and best_score >= threshold:
            clusters[best].items.append(item)
            _add_into(sums[best], vec)
        else:
            clusters.append(Cluster([item]))
            sums.append(dict(vec))

    clusters, sums = _merge_similar(clusters, sums, threshold)
    clusters.sort(key=lambda c: (-len(c.outlets), -len(c.items)))
    return clusters


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    denom = _norm(a) * _norm(b)
    if not denom:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(w * large[t] for t, w in small.items() if t in large) / denom


def _add_into(total: dict[str, float], vec: dict[str, float]) -> None:
    for term, weight in vec.items():
        total[term] = total.get(term, 0.0) + weight


def _merge_similar(clusters: list[Cluster], sums: list[dict[str, float]],
                   threshold: float, max_passes: int = 5):
    """Join groups whose combined profiles are similar.

    A single pass over the items depends on their order: two headlines about
    one event can start separate groups and only a third headline shows they
    belong together. Merging afterwards removes that order dependence.
    """
    for _ in range(max_passes):
        merged_any = False
        i = 0
        while i < len(clusters):
            best, best_score = -1, 0.0
            for j in range(i + 1, len(clusters)):
                score = _cosine(sums[i], sums[j])
                if score > best_score:
                    best, best_score = j, score
            if best >= 0 and best_score >= threshold:
                clusters[i].items.extend(clusters[best].items)
                _add_into(sums[i], sums[best])
                del clusters[best], sums[best]
                merged_any = True
            else:
                i += 1
        if not merged_any:
            break
    return clusters, sums
