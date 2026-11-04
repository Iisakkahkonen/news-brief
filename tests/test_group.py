from news_brief.group import Cluster, group_items, tokens

from conftest import make_item


def _ids(cluster):
    return sorted(i.item_id for i in cluster.items)


def test_same_event_from_different_outlets_is_grouped(day_of_news):
    clusters = group_items(day_of_news)
    groups = {tuple(_ids(c)) for c in clusters}

    assert ("geneva1", "geneva2", "geneva3") in groups
    assert ("flood1", "flood2") in groups


def test_unrelated_stories_stay_alone(day_of_news):
    clusters = group_items(day_of_news)
    singles = {c.items[0].item_id for c in clusters if len(c.items) == 1}
    assert singles == {"vote1", "tech1"}


def test_biggest_event_comes_first(day_of_news):
    clusters = group_items(day_of_news)
    assert _ids(clusters[0]) == ["geneva1", "geneva2", "geneva3"]
    assert len(clusters[0].outlets) == 3


def test_every_item_lands_in_exactly_one_group(day_of_news):
    clusters = group_items(day_of_news)
    all_ids = [i.item_id for c in clusters for i in c.items]
    assert sorted(all_ids) == sorted(i.item_id for i in day_of_news)


def test_threshold_controls_how_eager_grouping_is(day_of_news):
    strict = group_items(day_of_news, threshold=0.95)
    assert all(len(c.items) == 1 for c in strict)


def test_no_items_means_no_groups():
    assert group_items([]) == []


def test_lead_is_the_earliest_published_item(day_of_news):
    geneva = group_items(day_of_news)[0]
    assert geneva.lead.item_id == "geneva1"


def test_by_id_maps_ids_to_items():
    a = make_item("a", "X", "Alpha")
    b = make_item("b", "Y", "Beta")
    assert Cluster([a, b]).by_id() == {"a": a, "b": b}


def test_tokens_drop_stopwords_and_light_plurals():
    assert tokens("The talks were held in Geneva") == ["talk", "held", "geneva"]


def test_grouping_does_not_depend_on_item_order(day_of_news):
    import itertools
    expected = {tuple(_ids(c)) for c in group_items(day_of_news)}
    for order in itertools.islice(itertools.permutations(day_of_news), 0, 5040, 97):
        got = {tuple(_ids(c)) for c in group_items(list(order))}
        assert got == expected


def test_a_full_day_of_items_groups_quickly():
    import time
    words = [f"topic{n}" for n in range(60)]
    items = [
        make_item(f"i{n}", f"Outlet{n % 15}",
                  f"{words[n % 60]} {words[(n * 7) % 60]} update number {n}",
                  f"Report about {words[(n * 3) % 60]} and {words[(n * 11) % 60]}.")
        for n in range(410)
    ]
    start = time.perf_counter()
    clusters = group_items(items)
    assert time.perf_counter() - start < 5.0
    assert sum(len(c.items) for c in clusters) == 410


# --- splitting oversized groups (tuned on real headlines from 2026-11-04) -------

def _together(clusters, *needles):
    """True if, for each needle, some one cluster holds a title containing it
    and all needles share that cluster."""
    for cluster in clusters:
        titles = " || ".join(i.title for i in cluster.items)
        if all(n in titles for n in needles):
            return True
    return False


def test_midterms_blob_is_split_into_separate_stories(real_groups):
    from news_brief.group import MAX_CLUSTER_ITEMS, _split_if_large
    blob = Cluster(real_groups["midterms-blob"])
    assert len(blob.items) == 38

    parts = _split_if_large(blob, 0.30)

    assert len(parts) > 5
    assert all(len(p.items) <= MAX_CLUSTER_ITEMS for p in parts)
    assert sum(len(p.items) for p in parts) == 38
    # same story stays together ...
    assert _together(parts, "Talarico flips", "Lone Star turns blue")
    assert _together(parts, "says Iran war was the reason", "Vance: GOP lost midterms")
    assert _together(parts, "Who is Abdul El-Sayed", "'We did it!'")
    # ... different stories come apart
    assert not _together(parts, "Talarico flips", "says Iran war was the reason")
    assert not _together(parts, "Who is Abdul El-Sayed", "Lone Star turns blue")


def test_israel_blob_separates_the_fraud_claims_from_the_aid_story(real_groups):
    from news_brief.group import _split_if_large
    parts = _split_if_large(Cluster(real_groups["israel-blob"]), 0.30)

    assert _together(parts, "election commission rejects Netanyahu charges of fraud",
                     "rejects Benjamin Netanyahu charges of fraud")
    assert _together(parts, "Trump claims US will end military aid",
                     "welcomes Trump's plan to cut Israel aid")
    assert not _together(parts, "Trump claims US will end military aid",
                         "rejects Benjamin Netanyahu charges of fraud")


def test_small_groups_are_left_alone(day_of_news):
    from news_brief.group import _split_if_large
    cluster = Cluster(day_of_news[:3])
    assert _split_if_large(cluster, 0.30) == [cluster]


def test_a_group_that_cannot_be_told_apart_stays_whole():
    from news_brief.group import _split_if_large
    same = Cluster([make_item(f"s{n}", f"Outlet{n}", "Quake strikes coast", "Buildings fell.")
                    for n in range(15)])
    assert _split_if_large(same, 0.30) == [same]


# --- limiting what the model is shown ------------------------------------------

def test_for_prompt_returns_small_groups_unchanged(day_of_news):
    cluster = Cluster(day_of_news[:3])
    assert cluster.for_prompt(limit=12) is cluster


def test_for_prompt_takes_turns_across_outlets_earliest_first():
    items = [make_item(f"a{n}", "BigOutlet", f"Quake report {n}", hour=n) for n in range(8)]
    items += [make_item("b0", "Small1", "Quake report b", hour=20),
              make_item("c0", "Small2", "Quake report c", hour=21)]
    chosen = Cluster(items).for_prompt(limit=4)

    assert len(chosen.items) == 4
    assert {i.outlet for i in chosen.items} == {"BigOutlet", "Small1", "Small2"}
    assert [i.item_id for i in chosen.items if i.outlet == "BigOutlet"] == ["a0", "a1"]
