"""Probe whether a local language model remembers NBA game outcomes.

Asks the model about games before and after its checkpoint date, without any
pregame documents:
  * forced choice: log probability of each team's name as the winner, averaged
    over two prompts that name the teams in opposite orders;
  * recall: the final score it generates.
A model without game-level memory can still know which teams were strong, so the
decisive comparisons are upsets (the team with the worse season record won) and
exact final scores, before versus after the checkpoint. The forced choice has a
large home/position bias, so accuracy uses each period's median log odds as the
threshold, and AUC is reported alongside the AUC of hindsight season strength.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
GAMES_PATH = ROOT / "data/processed/seasonal/game_history.jsonl"
OUT_DIR = ROOT / "data/processed/memorization_probe"
REPORT_DIR = ROOT / "reports/memorization_probe"
SYSTEM = "You are a helpful assistant."
WINNER_PROMPTS = ("On {date}, the {away} played at the {home} in an NBA regular-season game. "
                  "Which team won? Answer with the team name only.",
                  "On {date}, the {home} hosted the {away} in an NBA regular-season game. "
                  "Which team won? Answer with the team name only.")
SCORE_PROMPT = ("On {date}, the {away} played at the {home} in an NBA regular-season game. "
                "What was the final score? Reply in exactly this format: "
                "<winning team> <points>, <losing team> <points>. If you are not sure, give your best guess.")
SCORE_PREFIX = "Final score:"


def read_rows(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def season_win_rates(games):
    wins, played = Counter(), Counter()
    for g in games:
        home_won = g["home_score"] > g["away_score"]
        for team, won in ((g["home_abbr"], home_won), (g["away_abbr"], not home_won)):
            wins[g["season"], team] += won
            played[g["season"], team] += 1
    return {key: wins[key] / played[key] for key in played}


def sample_games(games, seasons, per_season, seed):
    rng = random.Random(seed)
    chosen = []
    for season in seasons:
        pool = [g for g in games if g["season"] == season]
        chosen += sorted(rng.sample(pool, min(per_season, len(pool))), key=lambda g: g["scheduled_start_utc"])
    return chosen


def long_date(game_date_et):
    day = datetime.strptime(game_date_et, "%Y-%m-%d")
    return f"{day:%A}, {day:%B} {day.day}, {day.year}"


def nickname(team):
    return "Trail Blazers" if team.endswith("Trail Blazers") else team.split()[-1]


def chat_tokens(tokenizer, text, prefix=""):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(prompt + prefix, add_special_tokens=False)


def continuation_logprob(model, prompt, continuation):
    import mlx.core as mx
    tokens = prompt + continuation
    logits = model(mx.array(tokens)[None])[0, len(prompt) - 1:-1].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return sum(logprobs[i, t].item() for i, t in enumerate(continuation))


def parse_score(text, game):
    """Return (home points, away points) from '<team> <pts>, <team> <pts>', or None."""
    match = re.search(r"(.+?)\s+(\d{2,3})\s*[,\-–]\s*(.+?)\s+(\d{2,3})\b", text)
    if not match:
        return None
    home, away = nickname(game["home_team"]).lower(), nickname(game["away_team"]).lower()
    first, second = match.group(1).lower(), match.group(3).lower()
    a, b = int(match.group(2)), int(match.group(4))
    if home in first and away in second:
        return a, b
    if away in first and home in second:
        return b, a
    return None


def probe(model, tokenizer, game, max_tokens):
    from mlx_lm import generate
    fields = {"date": long_date(game["game_date_et"]), "home": game["home_team"], "away": game["away_team"]}
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    home, away = (tokenizer.encode(game[k], add_special_tokens=False) + end for k in ("home_team", "away_team"))
    log_odds = []
    for template in WINNER_PROMPTS:
        prompt = chat_tokens(tokenizer, template.format(**fields))
        log_odds.append(continuation_logprob(model, prompt, home) - continuation_logprob(model, prompt, away))
    mean = sum(log_odds) / len(log_odds)
    score_text = generate(model, tokenizer, prompt=chat_tokens(tokenizer, SCORE_PROMPT.format(**fields), SCORE_PREFIX),
                          max_tokens=max_tokens, verbose=False)
    return {"game_id": game["game_id"], "home_log_odds_by_prompt": log_odds,
            "home_probability": 1 / (1 + math.exp(-mean)), "score_text": score_text.strip(),
            "parsed_score": parse_score(score_text, game)}


def wilson(k, n, z=1.96):
    if not n:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 4), round(centre + half, 4)]


def rate(flags):
    flags = list(flags)
    return {"n": len(flags), "rate": round(sum(flags) / len(flags), 4) if flags else None,
            "ci95": wilson(sum(flags), len(flags))}


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label)
    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 4)


def log_odds(row):
    return sum(row["home_log_odds_by_prompt"]) / len(row["home_log_odds_by_prompt"])


def summarize(rows, games, strength):
    by_id = {g["game_id"]: g for g in games}
    groups = defaultdict(list)
    for r in rows:
        g = by_id[r["game_id"]]
        groups[str(g["season"])].append((r, g))
        groups[g["game_date_et"][:7]].append((r, g))
    summary = {}
    for key, pairs in sorted(groups.items()):
        correct, upset_correct, exact, margin3 = [], [], [], []
        scores = sorted(log_odds(r) for r, _ in pairs)
        median = scores[len(scores) // 2]
        labels, model_scores, strength_scores = [], [], []
        for r, g in pairs:
            home_won = g["home_score"] > g["away_score"]
            picks_home = log_odds(r) > median
            correct.append(picks_home == home_won)
            strength_gap = strength[g["season"], g["home_abbr"]] - strength[g["season"], g["away_abbr"]]
            labels.append(home_won)
            model_scores.append(log_odds(r))
            strength_scores.append(strength_gap)
            if (strength_gap >= 0) != home_won:
                upset_correct.append(picks_home == home_won)
            parsed = r["parsed_score"]
            exact.append(bool(parsed) and tuple(parsed) == (g["home_score"], g["away_score"]))
            margin3.append(bool(parsed) and abs((parsed[0] - parsed[1]) - (g["home_score"] - g["away_score"])) <= 3
                           and (parsed[0] > parsed[1]) == home_won)
        summary[key] = {"games": len(pairs), "model_auc": auc(model_scores, labels),
                        "hindsight_strength_auc": auc(strength_scores, labels),
                        "winner_accuracy_median_threshold": rate(correct), "upset_accuracy": rate(upset_correct),
                        "exact_score": rate(exact),
                        "winner_and_margin_within_3": rate(margin3),
                        "parsed_scores": sum(bool(r["parsed_score"]) for r, _ in pairs)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", required=True, help="short model name for output files")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--per-season", type=int, default=150)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()
    from mlx_lm import load

    games = read_rows(GAMES_PATH)
    chosen = sample_games(games, args.seasons, args.per_season, args.seed)
    out = OUT_DIR / args.label / "responses.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["game_id"]: r for r in read_rows(out)} if out.exists() else {}
    model, tokenizer = load(args.model_path)
    started = time.time()
    with open(out, "a") as handle:
        for i, game in enumerate(chosen):
            if game["game_id"] in done:
                continue
            row = probe(model, tokenizer, game, args.max_tokens)
            done[row["game_id"]] = row
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if i % 50 == 0:
                print(f"{i + 1}/{len(chosen)} {time.time() - started:.0f}s", flush=True)
    rows = [done[g["game_id"]] for g in chosen]
    report = {"model_path": args.model_path, "label": args.label, "seasons": args.seasons,
              "per_season": args.per_season, "seed": args.seed, "winner_prompts": WINNER_PROMPTS,
              "score_prompt": SCORE_PROMPT, "score_prefix": SCORE_PREFIX, "summary": summarize(rows, games, season_win_rates(games))}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    for season in map(str, args.seasons):
        s = report["summary"][season]
        print(season, "AUC", s["model_auc"], "vs strength", s["hindsight_strength_auc"],
              "winner", s["winner_accuracy_median_threshold"]["rate"], "upsets", s["upset_accuracy"]["rate"],
              f"(n={s['upset_accuracy']['n']})", "exact score", s["exact_score"]["rate"],
              "margin<=3", s["winner_and_margin_within_3"]["rate"])


if __name__ == "__main__":
    main()
