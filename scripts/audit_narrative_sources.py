"""Audit discovered NBA team articles without promoting any narrative to inputs."""
import argparse
from collections import Counter
from datetime import datetime, timedelta
from html.parser import HTMLParser
import json
import re

from collect_pilot import Cache, ROOT, TextOnly, iso, stamp, write_json, write_jsonl


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.links = [], []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.current = {"id": attrs.get("id"), "type": attrs.get("type"), "text": ""}
        elif tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.scripts.append(self.current)
            self.current = None


def parse_article(body, *, article_types=("NewsArticle",)):
    parser = PageParser()
    parser.feed(body.decode("utf-8"))
    ld_articles, pages = [], []
    for script in parser.scripts:
        if script["id"] == "__NEXT_DATA__":
            pages.append(json.loads(script["text"])["props"]["pageProps"]["pageObject"])
        elif script["type"] == "application/ld+json":
            value = json.loads(script["text"])
            nodes = value if isinstance(value, list) else value.get("@graph", [value])
            ld_articles.extend(node for node in nodes if node.get("@type") in article_types)
    if len(ld_articles) != 1 or len(pages) != 1 or pages[0].get("type") != "article":
        raise ValueError("Expected one allowed article schema and one article pageObject")
    ld, page = ld_articles[0], pages[0]
    # Use only the article's own blocks. Never flatten navigation, related news,
    # page layout, or the entire hydration object into a model-ready document.
    blocks = []
    for block in page.get("contentStructured", []):
        if block.get("type") in ("paragraph", "heading", "list"):
            text = TextOnly()
            text.feed(block.get("html") or "")
            value = " ".join(" ".join(text.parts).split())
            if value:
                blocks.append(value)
    return {"title": ld.get("headline"), "cms_title": page.get("title"),
            "published": ld.get("datePublished"), "modified": ld.get("dateModified"),
            "cms_published": page.get("date"), "cms_modified": page.get("modified"),
            "url": ld.get("mainEntityOfPage", {}).get("@id"), "cms_url": page.get("permalink"),
            "opening": blocks[0] if blocks else "",
            "header": " ".join(blocks[:2]), "text": "\n\n".join(blocks)}


def article_checks(article, game, cutoff, url):
    reasons = []
    if article["url"] != url or article["cms_url"] != url:
        reasons.append("url_mismatch")
    if article["title"] != article["cms_title"]:
        reasons.append("title_metadata_disagrees")
    times = {}
    for field in ("published", "modified", "cms_published", "cms_modified"):
        try:
            value = stamp(article[field])
            if value.tzinfo is None:
                raise ValueError("Timestamp lacks timezone")
            times[field] = value
        except (ValueError, TypeError, AttributeError):
            reasons.append("missing_or_invalid_" + field)
    if len(times) == 4:
        if times["published"] != times["cms_published"] or times["modified"] != times["cms_modified"]:
            reasons.append("timestamp_metadata_disagrees")
        if any(value > cutoff for value in times.values()):
            reasons.append("after_cutoff")
        if times["modified"] < times["published"] or times["cms_modified"] < times["cms_published"]:
            reasons.append("modified_before_published")
    title, opening = (article["title"] or "").lower(), article["opening"].lower()
    for side in ("away", "home"):
        nickname = game[side + "_team"].split()[-1].lower()
        if nickname not in title or nickname not in opening:
            reasons.append(side + "_team_missing_from_title_or_opening")
    day = datetime.fromisoformat(game["game_date_et"])
    date_pattern = rf'\b(?:{day:%B}|{day:%b}\.?)\s+0?{day.day},?\s+{day.year}\b'
    if not re.search(date_pattern, (article["title"] or "") + " " + article["header"], re.I):
        reasons.append("game_date_missing_from_header")
    if not article["text"]:
        reasons.append("empty_article")
    return reasons


def audit_article(cache, spec, game, cutoff_minutes, directory):
    cutoff = stamp(game["scheduled_start_utc"]) - timedelta(minutes=cutoff_minutes)
    body, meta = cache.get(spec["url"], "narrative_pages", ".html")
    row = dict(spec, game_id=game["game_id"], cutoff_utc=iso(cutoff), source=meta,
               included_in_inputs=False, decision="fetch_failed")
    if body is not None:
        try:
            article = parse_article(body)
            reasons = article_checks(article, game, cutoff, spec["url"])
            path = directory / "narrative_candidates" / f'{game["espn_id"]}-{spec["team"]}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(article["text"] + "\n")
            row.update(title=article["title"], published=article["published"], modified=article["modified"],
                       cms_published=article["cms_published"], cms_modified=article["cms_modified"],
                       text_path=str(path.relative_to(ROOT)), rejection_reasons=reasons,
                       decision="rejected" if reasons else "candidate_requires_content_and_archive_review",
                       publication_evidence="Retrospective publisher metadata, not an independently archived pregame snapshot")
        except (ValueError, KeyError, TypeError) as exc:
            row.update(decision="parse_error", error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/narrative_sources.json").read_text())
    history = {}
    for name in ("game_history.jsonl", "seasonal/game_history.jsonl"):
        path = ROOT / "data/processed" / name
        if path.exists():
            history.update((g["game_id"], g) for g in map(json.loads, path.read_text().splitlines()))
    # Resolve all schedule joins before touching outputs; the source examples
    # need the completed seasonal schedule history, not only the pilot manifest.
    matched = []
    for spec in config["articles"]:
        matches = [g for g in history.values() if all(g[k] == spec[k] for k in ("game_date_et", "away_abbr", "home_abbr"))]
        if len(matches) != 1:
            raise ValueError(f"Expected one schedule match for {spec['url']}; run seasonal collection first")
        matched.append(matches[0])
    directory, report_dir = ROOT / "data/processed/narrative_sources", ROOT / "reports/narrative_sources"
    cache = Cache(offline=args.offline)
    articles = [audit_article(cache, spec, game, config["cutoff_minutes"], directory)
                for spec, game in zip(config["articles"], matched)]
    indexes = []
    for spec in config["discovery_pages"]:
        body, meta = cache.get(spec["url"], "narrative_pages", ".html")
        parser = PageParser()
        if body is not None:
            parser.feed(body.decode("utf-8"))
        links = sorted({link for link in parser.links if re.match(r"https://www\.nba\.com/gamenotes/[^/]+\.pdf(?:\?.*)?$", link)})
        indexes.append(dict(spec, source=meta, rolling_pdf_links=links, included_in_inputs=False,
                            decision="rolling_links_only_no_historical_coverage_claim" if links else "no_rolling_links_found"))
    write_jsonl(directory / "audit.jsonl", articles)
    summary = {"scope": config["scope"], "articles_attempted": len(articles),
               "article_decisions": dict(Counter(row["decision"] for row in articles)),
               "narrative_inputs_approved": 0, "articles": articles, "discovery_pages": indexes}
    write_json(report_dir / "coverage.json", summary)
    lines = ["# Additional narrative source audit", "", config["scope"] + ".", "",
             "No article is approved as model input. Retrospectively retrieved publisher timestamps are evidence to investigate, not proof of an unchanged pregame version.", "",
             "| Team | Game | Published (UTC) | Modified (UTC) | Cutoff (UTC) | Decision |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in articles:
        details = ", ".join(row.get("rejection_reasons", [])) or row["decision"]
        lines.append(f'| {row["team"]} | [{row["game_date_et"]} {row["away_abbr"]}@{row["home_abbr"]}]({row["url"]}) | {row.get("published", "unknown")} | {row.get("modified", "unknown")} | {row["cutoff_utc"]} | {details} |')
    for row in articles:
        if row.get("review_note"):
            lines.extend(["", f'{row["team"]} {row["game_date_et"]}: {row["review_note"]}'])
    lines.extend(["", "## Archive discovery", ""])
    for row in indexes:
        lines.append(f'- [{row["team"]} index]({row["url"]}): {row["review_note"]} Found {len(row["rolling_pdf_links"])} rolling PDF links.')
    lines.extend(["", "## Next step", "",
                  "Expand the promising dated article source with a schedule-defined sample and independent pregame snapshot evidence. Review article-only text before promotion; keep rejected revisions quarantined. Rolling PDFs are candidates for prospective capture, not historical backfill.", "",
                  "Rebuild: `.venv/bin/python scripts/audit_narrative_sources.py --offline` after collecting the seasonal schedules.", ""])
    (report_dir / "findings.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("articles", "discovery_pages")}, indent=2))


if __name__ == "__main__":
    main()
