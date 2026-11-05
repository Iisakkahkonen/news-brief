import pytest

from news_brief.summarise import Sentence
from news_brief.verify import magnitudes_in, names_in, numbers_in, verify, verify_sentence

from conftest import make_item

FLOOD = make_item(
    "flood1", "Dawn", "Floods kill 14 in northern Pakistan",
    "Heavy rain triggered landslides in Swat. Officials said 3,200 people were displaced.",
)
FLOOD2 = make_item(
    "flood2", "The Hindu", "Pakistan floods: death toll reaches 14",
    "The National Disaster Management Authority said two bridges collapsed.",
)
OTHER = make_item("other1", "NPR", "Voters in Brazil head to the polls", "Turnout was steady.")
ALLOWED = {i.item_id: i for i in (FLOOD, FLOOD2)}


def check(text, *sources):
    return verify_sentence(Sentence(text, tuple(sources)), ALLOWED)


# --- sentences that must pass -------------------------------------------------

def test_sentence_supported_by_its_source_passes():
    assert check("Floods killed 14 people in northern Pakistan.", "flood1") is None


def test_number_formatting_differences_are_accepted():
    assert check("About 3200 people were displaced in Swat.", "flood1") is None


def test_number_word_matches_digit_in_source():
    assert check("Two bridges collapsed, the National Disaster Management Authority said.",
                 "flood2") is None
    assert check("Fourteen people died in Pakistan.", "flood1") is None


def test_digit_in_sentence_matches_number_word_in_source():
    assert check("2 bridges collapsed in Pakistan.", "flood2") is None


def test_ending_changes_of_a_name_are_accepted():
    assert check("Pakistani officials said 3,200 people were displaced.", "flood1") is None


def test_sources_can_be_combined_across_items():
    assert check("The National Disaster Management Authority said two bridges "
                 "collapsed as floods killed 14 in Pakistan.", "flood1", "flood2") is None


def test_sentence_opening_function_words_are_not_treated_as_names():
    assert check("The floods killed 14 people. ".strip(), "flood1") is None
    assert check("According to officials, 3,200 people were displaced.", "flood1") is None


# --- deliberately bad model output: all of these must be rejected --------------

def test_sentence_without_sources_is_rejected():
    assert check("Floods killed 14 people in northern Pakistan.") == "no sources listed"


def test_made_up_source_id_is_rejected():
    reason = check("Floods killed 14 people in northern Pakistan.", "flood1", "invented99")
    assert reason == "source id not among this event's items: invented99"


def test_source_from_a_different_event_is_rejected():
    assert "not among this event's items" in check("Turnout was steady.", "other1")


def test_invented_number_is_rejected():
    reason = check("Floods killed 140 people in northern Pakistan.", "flood1")
    assert reason == "number not in cited items: 140"


def test_number_taken_from_the_wrong_item_is_rejected():
    # 3,200 appears only in flood1; the sentence cites flood2.
    assert check("3,200 people were displaced.", "flood2") == "number not in cited items: 3200"


def test_invented_name_is_rejected():
    reason = check("Floods killed 14 people in northern Pakistan, Prime Minister Sharif said.",
                   "flood1")
    assert reason.startswith("name not in cited items:")
    assert "sharif" in reason


def test_invented_place_is_rejected():
    reason = check("Floods killed 14 in northern Pakistan and Afghanistan.", "flood1")
    assert reason == "name not in cited items: afghanistan"


def test_invented_plural_quantity_word_is_rejected():
    reason = check("Floods displaced 3,200 people and caused losses of millions.", "flood1")
    assert reason == "quantity word not in cited items: million"


def test_invented_magnitude_is_rejected():
    assert check("Floods displaced 3,200 million people.", "flood1") == \
        "quantity word not in cited items: million"


@pytest.mark.parametrize("text, missing", [
    ("Floods killed 14 people in northern Pakistan on Tuesday.", "tuesday"),
    ("Floods killed 14 people in northern Pakistan in March.", "march"),
    ("Yesterday floods killed 14 people in northern Pakistan.", "yesterday"),
])
def test_invented_dates_and_time_words_are_rejected(text, missing):
    assert check(text, "flood1") == f"name not in cited items: {missing}"


def test_a_date_that_is_in_the_source_is_accepted():
    allowed = {"d": make_item("d", "BBC", "Talks to resume on Tuesday", "Mediators confirmed.")}
    assert verify_sentence(Sentence("Talks will resume on Tuesday.", ("d",)), allowed) is None


def test_empty_sentence_is_rejected():
    assert check("   ", "flood1") == "empty sentence"


def test_similar_but_different_short_names_are_not_confused():
    # "Iran" must not be accepted because the source says "Iraq" (short words need exact match).
    allowed = {"x": make_item("x", "BBC", "Strikes hit Iraq", "Reports of damage.")}
    assert verify_sentence(Sentence("Strikes hit Iran.", ("x",)), allowed) == \
        "name not in cited items: iran"


# --- verify(): splitting and reporting -----------------------------------------

def test_verify_returns_accepted_and_rejected_with_reasons():
    good = Sentence("Floods killed 14 people in northern Pakistan.", ("flood1",))
    bad_number = Sentence("Floods killed 99 people in northern Pakistan.", ("flood1",))
    bad_id = Sentence("Floods hit Pakistan.", ("nope",))

    accepted, rejected = verify([good, bad_number, bad_id], ALLOWED)

    assert accepted == [good]
    assert [r.sentence for r in rejected] == [bad_number, bad_id]
    assert rejected[0].reason == "number not in cited items: 99"
    assert "nope" in rejected[1].reason


def test_verify_with_nothing_to_check():
    assert verify([], ALLOWED) == ([], [])


# --- helpers -------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("14 died and 3,200 fled", {"14", "3200"}),
    ("Two of the 12 bridges, 4.5 km long", {"2", "12", "4.5"}),
    ("no figures here", set()),
])
def test_numbers_in(text, expected):
    assert numbers_in(text) == expected


def test_magnitudes_in():
    assert magnitudes_in("Two million people and 3 thousand homes") == {"million", "thousand"}


def test_names_in_skips_function_words_and_lowercase():
    assert names_in("The army of Israel's ally said Gaza was hit.") == ["israel", "gaza"]


def test_names_in_skips_number_words_that_start_a_sentence():
    assert names_in("Fourteen people died and Two million fled Gaza.") == ["gaza"]


# --- found in the first real check log (2026-11-05): false alarms and one true catch ----

ARAB = make_item("ar", "Anadolu Agency (World)",
                 "Arab party recommends opposition leader Eisenkot to form Israeli government",
                 "Israel's Arab-majority lists won 19 seats after the election.")
GUARD = make_item("gu", "The Guardian (World)", "Wong sentenced to 10 years",
                  "The 30-year-old pleaded guilty in September.")
TRUMP = make_item("tr", "NPR (World)", "Trump plans to end US military aid to Israel",
                  "Critics in the U.S. said Netanyahu pulled Trump into the war.")
REAL = {i.item_id: i for i in (ARAB, GUARD, TRUMP)}


def real(text, *sources):
    return verify_sentence(Sentence(text, tuple(sources)), REAL)


def test_hyphenated_compound_is_checked_by_its_capitalised_part():
    assert real("Following the election, Arab-majority lists won 19 seats.", "ar") is None


def test_only_the_capitalised_part_of_a_hyphenated_word_is_a_name():
    assert names_in("Arab-majority lists and El-Sayed") == ["arab", "el", "sayed"]


def test_hyphenated_name_still_has_to_be_in_the_source():
    assert real("Kurdish-majority lists won 19 seats.", "ar") == \
        "name not in cited items: kurdish"


def test_dotted_abbreviations_match_the_source():
    assert real("Critics in the U.S. said Netanyahu pulled Trump into the war.", "tr") is None
    assert real("Critics in the US said Netanyahu pulled Trump into the war.", "tr") is None
    assert names_in("U.S.-Israel ties and the U.K.") == ["us", "israel", "uk"]


def test_attributing_to_an_outlet_is_allowed_when_that_outlet_is_cited():
    assert real("According to The Guardian, the 30-year-old pleaded guilty in September.",
                "gu") is None


def test_attributing_to_an_outlet_that_was_not_cited_is_rejected():
    assert real("According to Al Jazeera, the 30-year-old pleaded guilty in September.",
                "gu") == "name not in cited items: al, jazeera"


def test_outlet_label_drops_the_feed_section():
    from news_brief.verify import outlet_label
    assert outlet_label("The Guardian (World)") == "The Guardian"
    assert outlet_label("Al Jazeera (All)") == "Al Jazeera"
    assert outlet_label("Times of Israel") == "Times of Israel"


def test_a_first_name_the_source_does_not_contain_is_still_rejected():
    # "Benjamin Netanyahu" when the items only say "Netanyahu": knowledge from outside.
    assert real("Critics said Benjamin Netanyahu pulled Trump into the war.", "tr") == \
        "name not in cited items: benjamin"
