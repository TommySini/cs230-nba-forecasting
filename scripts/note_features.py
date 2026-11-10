"""Turn the admitted pregame game notes into numeric features for the boosting model, without an LLM.

Covers every admitted note (notes.parquet; all are public at or before the game's cutoff, which is
re-checked here). Each note is written by one of the two teams, so features belong to that side; the
model gets home, away and home-minus-away columns, missing where a team has no note. The notes are a
selected corpus (some teams and seasons), not league-wide coverage.

Families, per publishing team (several notes for one game are averaged):
  meta       note found, log length, pages, PDF vs web post
  sentiment  VADER compound mean and positive/negative sentence shares over prose sentences
  lexicon    rates per 1,000 words of a basketball word list: winning momentum, losing momentum,
             injury/absence, returns, roster moves, coaching changes
  numbers    current streak (+wins / -losses), last-10 win share, season-series margin, "won X of last Y"
  status     players marked out / questionable / probable inside injury or status-report sections
  topics     10 LSA components of TF-IDF (vocabulary and components fitted on TRAIN notes only)
  text_score log-odds that the publishing team wins, from logistic regression on TF-IDF. TRAIN notes
             are scored out of fold with folds grouped by season; DEV/TEST notes by the model fitted on
             all TRAIN notes. Only TRAIN outcomes are read.
"""
from collections import defaultdict
import math
import re

import numpy as np

SIDES = ("home", "away")
TOPICS = 10
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12}
NUMBER = r"(\d{1,2}|" + "|".join(WORDS) + r")"
STATUS_HEADING = re.compile(r"(?i)injur(?:y|ies)\s*(?:/\s*status\s*)?(?:report|update|list)|status\s+report")
STATUS_WINDOW = 900
STATUS_WORD = re.compile(r"(?i)\b(out|doubtful|questionable|probable|day[- ]to[- ]day|gtd)\b")
G_LEAGUE = re.compile(r"(?i)g[- ]league|two[- ]way|on assignment")
LEXICON = {
    "winning": r"winning streak|win streak|won \w+ straight|won \w+ consecutive|season[- ]high|career[- ]high|"
               r"victor(?:y|ies)|dominan\w*|hot streak|rolling|surg(?:e|ing)",
    "losing": r"losing streak|skid|lost \w+ straight|lost \w+ consecutive|slump|struggl\w+|defeats?|cold streak|"
              r"season[- ]low|swoon",
    "injury": r"injur\w*|sprain\w*|strain\w*|surgery|sidelined|torn|fractur\w*|concussion|soreness|illness|"
              r"inactive|will not play|did not play|dnp|missed \w+ games?|out indefinitely|rehab\w*",
    "returns": r"\breturn(?:s|ed|ing)?\b|back in the lineup|cleared to|available|season debut|\bdebut\w*",
    "roster": r"\btrade[ds]?\b|acquired|\bsigned\b|waived|10-day|ten-day|claimed off|converted",
    "coaching": r"interim|head coach|fired|dismissed|assistant coach",
}
LEXICON = {k: re.compile(r"(?i)\b(?:" + v + r")") for k, v in LEXICON.items()}
STREAK = [re.compile(r"(?i)\b(won|lost)\s+" + NUMBER + r"\s+(?:straight|consecutive|in a row)"),
          re.compile(r"(?i)\b" + NUMBER + r"[- ]game\s+(winning|losing)\s+(?:streak|skid)")]
LAST_TEN = re.compile(r"(?i)\b(\d{1,2})-(\d{1,2})\s+(?:in|over)\s+(?:their|its|the)\s+(?:last|past)\s+(?:10|ten)\b")
WON_OF = re.compile(r"(?i)\bwon\s+" + NUMBER + r"\s+of\s+(?:their|its|the)\s+(?:last|past)\s+" + NUMBER + r"\b")
SERIES = re.compile(r"(?i)\b(lead|leads|trail|trails)\s+the\s+(?:\d{4}-\d{2}\s+)?season\s+series,?\s+(\d)-(\d)")

PER_TEAM = (("note_found", "note_log_chars", "note_pages", "note_pdf")
            + ("sentiment_compound", "sentiment_positive", "sentiment_negative")
            + tuple(f"lexicon_{k}" for k in LEXICON)
            + ("streak", "last10_win_share", "series_margin", "won_of_last_share")
            + ("status_out", "status_questionable", "status_probable")
            + tuple(f"topic_{i}" for i in range(TOPICS)) + ("text_score",))
FAMILIES = {
    "meta": ("note_found", "note_log_chars", "note_pages", "note_pdf"),
    "sentiment": ("sentiment_compound", "sentiment_positive", "sentiment_negative"),
    "lexicon": tuple(f"lexicon_{k}" for k in LEXICON),
    "numbers": ("streak", "last10_win_share", "series_margin", "won_of_last_share"),
    "status": ("status_out", "status_questionable", "status_probable"),
    "topics": tuple(f"topic_{i}" for i in range(TOPICS)),
    "text_score": ("text_score",),
}
FEATURE_NAMES = tuple(f"{side}_{name}" for side in ("home", "away", "diff") for name in PER_TEAM)


def feature_names(families):
    keep = {name for family in families for name in FAMILIES[family]}
    return [f"{side}_{name}" for side in ("home", "away", "diff") for name in PER_TEAM
            if name in keep]


def _number(token):
    return int(token) if token.isdigit() else WORDS[token.lower()]


def prose_sentences(text):
    """Sentences that read as prose, not table rows: enough words, mostly letters."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        words = sentence.split()
        letters = sum(c.isalpha() for c in sentence)
        if len(words) >= 6 and letters >= 0.6 * max(1, len(sentence.replace(" ", ""))):
            out.append(sentence.strip())
    return out


def status_counts(text):
    """Status words inside injury/status-report sections, skipping G League clauses."""
    counts = {"out": 0, "questionable": 0, "probable": 0}
    for heading in STATUS_HEADING.finditer(text):
        section = text[heading.end():heading.end() + STATUS_WINDOW]
        for clause in re.split(r"[\n;]|(?<=[a-z)])\.\s", section):
            if G_LEAGUE.search(clause):
                continue
            for word in STATUS_WORD.findall(clause):
                word = word.lower()
                counts["out" if word == "out" else "probable" if word == "probable" else "questionable"] += 1
    return counts


def numbers(text):
    """First stated streak, last-10 share, season-series margin and 'won X of last Y' share (NaN if absent)."""
    out = dict.fromkeys(("streak", "last10_win_share", "series_margin", "won_of_last_share"), math.nan)
    found = []
    for pattern in STREAK:
        for m in pattern.finditer(text):
            a, b = m.group(1), m.group(2)
            if a.lower() in ("won", "lost"):
                found.append((m.start(), _number(b) * (1 if a.lower() == "won" else -1)))
            else:
                found.append((m.start(), _number(a) * (1 if b.lower() == "winning" else -1)))
    if found:
        out["streak"] = float(min(found)[1])
    m = LAST_TEN.search(text)
    if m and int(m.group(1)) + int(m.group(2)) == 10:
        out["last10_win_share"] = int(m.group(1)) / 10
    m = SERIES.search(text)
    if m:
        lead = m.group(1).lower().startswith("lead")
        a, b = int(m.group(2)), int(m.group(3))
        out["series_margin"] = float(abs(a - b) if lead else -abs(a - b))
    m = WON_OF.search(text)
    if m and 0 < _number(m.group(2)) <= 20 and _number(m.group(1)) <= _number(m.group(2)):
        out["won_of_last_share"] = _number(m.group(1)) / _number(m.group(2))
    return out


def document_features(text, pages, source_kind, analyzer):
    words = max(1, len(text.split()))
    sentences = prose_sentences(text)
    scores = [analyzer.polarity_scores(s)["compound"] for s in sentences]
    row = {"note_found": 1.0, "note_log_chars": math.log1p(len(text)), "note_pages": float(pages),
           "note_pdf": float(source_kind == "pdf"),
           "sentiment_compound": float(np.mean(scores)) if scores else math.nan,
           "sentiment_positive": float(np.mean([s > .05 for s in scores])) if scores else math.nan,
           "sentiment_negative": float(np.mean([s < -.05 for s in scores])) if scores else math.nan}
    row.update({f"lexicon_{k}": 1000 * len(p.findall(text)) / words for k, p in LEXICON.items()})
    row.update(numbers(text))
    row.update({f"status_{k}": float(v) for k, v in status_counts(text).items()})
    return row


def text_models(docs, train_outcome, seed=230):
    """LSA topics and out-of-fold text score per document. docs: list of dicts with text, split, season, won."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [d["text"] for d in docs]
    train = np.array([d["split"] == "train" for d in docs])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=.9, max_features=50000, sublinear_tf=True,
                                 stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b")
    x_train = vectorizer.fit_transform([t for t, k in zip(texts, train) if k])
    x_all = vectorizer.transform(texts)
    svd = TruncatedSVD(TOPICS, random_state=seed).fit(x_train)
    topics = svd.transform(x_all)

    y = np.array([train_outcome.get(i, -1) for i in range(len(docs))])
    labelled = train & (y >= 0)
    seasons = np.array([d["season"] for d in docs])
    groups = sorted(set(seasons[labelled]))
    fold_of = {s: i % 5 for i, s in enumerate(groups)}
    score = np.full(len(docs), np.nan)
    model = lambda: LogisticRegression(C=1.0, max_iter=2000)
    for fold in range(5):
        held = labelled & np.array([fold_of.get(s, -1) == fold for s in seasons])
        fit = labelled & ~held
        if held.any() and len(set(y[fit])) == 2:
            score[held] = model().fit(x_all[fit], y[fit]).decision_function(x_all[held])
    final = model().fit(x_all[labelled], y[labelled])
    score[~train] = final.decision_function(x_all[~train])
    return topics, score


def build(dataset, splits=("train", "dev", "test")):
    """game_id -> {feature: value} for every game with an admitted note in `splits`."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    from bam_data import read_snapshot_inputs
    from collect_pilot import stamp
    from evaluate_notes_orthogonality import page_texts

    texts = page_texts(dataset.directory)
    meta = {r["sha256"]: r for r in dataset.table("notes")}
    pages = defaultdict(int)
    for r in dataset.table("note_pages"):
        pages[r["source_sha256"]] += 1
    outcome = {r["game_id"]: r["home_win"] for r in dataset.labels(("train",), eligible_only=False)}
    analyzer = SentimentIntensityAnalyzer()
    docs = []
    for game in read_snapshot_inputs(dataset.directory, dataset.root, splits=splits):
        for source in game["sources"]:
            if stamp(source["archive_time"]) > stamp(game["cutoff_utc"]):
                raise ValueError(f"Note not public before its cutoff: {game['game_id']}")
            side = "home" if source["publisher"] == game["home_abbr"] else "away" if source["publisher"] == game["away_abbr"] else None
            if side is None:
                continue
            text = texts[source["sha256"]]
            doc = {"game_id": game["game_id"], "side": side, "split": game["split"], "season": game["season"], "text": text}
            doc.update(document_features(text, pages[source["sha256"]], meta[source["sha256"]]["source_kind"], analyzer))
            docs.append(doc)
    won = {i: int(outcome[d["game_id"]] == (d["side"] == "home")) for i, d in enumerate(docs)
           if d["split"] == "train" and d["game_id"] in outcome}
    topics, score = text_models(docs, won)
    for i, d in enumerate(docs):
        d.update({f"topic_{k}": float(topics[i, k]) for k in range(TOPICS)}, text_score=float(score[i]))

    by_side = defaultdict(lambda: defaultdict(list))
    for d in docs:
        by_side[d["game_id"]][d["side"]].append(d)
    features = {}
    for game_id, sides in by_side.items():
        row = {}
        for side in SIDES:
            group = sides.get(side, [])
            for name in PER_TEAM:
                values = [g[name] for g in group if not math.isnan(g[name])]
                row[f"{side}_{name}"] = float(np.mean(values)) if values else (0.0 if name == "note_found" else math.nan)
        for name in PER_TEAM:
            row[f"diff_{name}"] = row[f"home_{name}"] - row[f"away_{name}"]
        features[game_id] = row
    return features, docs
