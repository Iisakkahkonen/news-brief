# news-brief

A short, daily world-events brief written by an AI model from the headlines of
15 news outlets, where **code removes any sentence it cannot trace to a real
source**. The point of the project is not the summary; it is the check that
stops the model from inventing things.

Status: work in progress. Feed reading, grouping, summarising and the source
check are built and tested. Email delivery, the web archive and the daily
schedule come next.

## How it works

1. **Collect** - read the RSS/Atom feeds listed in `config/outlets.toml`. Only
   the headline and the short summary each outlet publishes are used; the
   brief links out to the full articles.
2. **Group** - items about the same event are clustered by text similarity
   (`group.py`). Very large groups are re-clustered so one big news story does
   not swallow unrelated ones.
3. **Summarise** - the model writes 1-4 neutral sentences per event, and must
   list the id of each item every sentence is based on (`summarise.py`).
4. **Check** - plain code, no AI (`verify.py`). A sentence is dropped unless:
   it lists sources; every source is one of that event's items; every number
   and every capitalised word (names, places, dates) appears in the cited
   items. Dropped sentences are written to a log, never sent.
5. **Render** - plain text with numbered, credited sources.

What the check does **not** prove: that a sentence is true, or that the cited
item supports its meaning. It proves the specifics (who, where, how many) came
from the cited text and not from the model's imagination. It errs on the side
of dropping good sentences (for example "2 million" will not match
"2,000,000").

### Wording rules given to the model

- Use only the numbered items; add no outside knowledge, names, numbers or dates.
- Neutral tone. Where outlets disagree, say who said what.
- Report claims as claims ("said", "according to"); never state a disputed
  claim as settled fact.

## Sources

Headlines and summaries come from these outlets' public feeds. Each brief
credits the outlet and links to the original item for every sentence.

[BBC News](https://www.bbc.com/news/world) -
[Deutsche Welle](https://www.dw.com/en/) -
[Al Jazeera](https://www.aljazeera.com/) -
[The Guardian](https://www.theguardian.com/world) -
[NPR](https://www.npr.org/sections/world/) -
[France 24](https://www.france24.com/en/) -
[Euronews](https://www.euronews.com/) -
[UN News](https://news.un.org/) -
[ReliefWeb](https://reliefweb.int/) -
[Anadolu Agency](https://www.aa.com.tr/en) -
[The Times of Israel](https://www.timesofisrael.com/) -
[Dawn](https://www.dawn.com/) -
[The Hindu](https://www.thehindu.com/news/international/) -
[allAfrica](https://allafrica.com/) -
[Yle News](https://yle.fi/news)

The outlets differ in ownership and viewpoint on purpose. Feed URLs and the
approval status of each outlet are in `config/outlets.toml`.

## Run it

Python 3.11 or newer. No packages are needed to run it (the tests need pytest).

```
pip install -e ".[dev]"
python -m pytest                        # run the tests
python -m news_brief                    # fetch the feeds, count items per outlet
python -m news_brief --show-groups      # also print how items group into events
python -m news_brief --brief            # write the AI brief (needs a Gemini key)
```

`--brief` writes `out/brief-<date-time>.txt` and `out/check-log-<date-time>.txt` (what
the source check removed, and why).

### Gemini API key (free)

The brief is written by Google's Gemini API on its free tier.

1. Open Google AI Studio's API keys page and choose **Create API key**. Copy
   the key and keep it private: never put it in the repo, in a chat, or in a
   URL.
2. Do **not** link a billing account to the key's Google Cloud project. Google's
   documentation says a key inherits the billing status of its project, so a
   project without billing has nothing to charge.
3. In PowerShell, in the same window you run the command from:

   ```
   $env:GEMINI_API_KEY = "paste-your-key-here"
   python -m news_brief --brief
   ```

Safeguards in the code: at most `--max-calls` model calls per run (default 15);
calls are spaced out; a failed call is retried once; a quota, permission or
payment error stops the run instead of retrying; the key is sent in a request
header and removed from any error text; Google is asked not to store prompts.

## Project layout

```
config/outlets.toml     which feeds are read, and whether each is approved
src/news_brief/
  feeds.py              RSS 2.0 / RSS 1.0 / Atom reader
  group.py              cluster items into events
  summarise.py          prompt and strict parser for the model's answer
  verify.py             the source check
  pipeline.py           ties the steps together, renders the brief
  gemini.py             Gemini API client
  model.py              the one-method model interface, call budget
  run.py                writes the brief and the check log
tests/                  tests for every step, including deliberately bad model output
```

## Licence

Apache 2.0, see `LICENSE`.
