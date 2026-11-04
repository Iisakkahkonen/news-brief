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
