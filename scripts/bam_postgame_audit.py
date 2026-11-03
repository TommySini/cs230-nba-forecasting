"""Metadata timing audit for normalized TRAIN postgames at research fold boundaries."""
from bam_data import stamp


def audit_recap_fold_times(inputs, postgames, folds=3):
    """Audit metadata at actual expanding research folds, without archive claims."""
    train = {row["game_id"]: row for row in inputs if row["split"] == "train"}
    dates = sorted({row["game_date_et"] for row in train.values()})
    if not dates or folds < 1:
        raise ValueError("Need TRAIN dates and positive research fold count")
    records = {}
    for record in postgames:
        if record.get("split") != "train" or record.get("normalized") is not True or record["game_id"] not in train:
            raise ValueError("Fold audit accepts only normalized TRAIN postgames")
        if record["game_id"] in records:
            raise ValueError("Duplicate postgame record")
        records[record["game_id"]] = record
    boundaries = []
    blocks = min(folds, len(dates))
    for number in range(blocks):
        first = dates[number * len(dates) // blocks]
        predictions = [row for row in train.values() if row["game_date_et"] == first]
        boundaries.append((number, first, min(stamp(row["cutoff_utc"]) for row in predictions)))
    heldout = [row for row in inputs if row["split"] != "train"]
    if heldout:
        first = min(row["game_date_et"] for row in heldout)
        boundaries.append(("final", first, min(stamp(row["cutoff_utc"]) for row in heldout if row["game_date_et"] == first)))
    results = []
    for number, first, boundary in boundaries:
        fitting = [gid for gid, row in train.items() if row["game_date_et"] < first]
        missing, late, checked, timestamps = [], [], [], []
        for gid in fitting:
            for source in records.get(gid, {}).get("sources", []):
                if "recap" not in source.get("content_types", []):
                    continue
                fields = {}
                for page in source["pages"]:
                    for line in page["lines"]:
                        for field in ("published", "lastModified"):
                            if line.get("source_pointer", "").endswith("/article/" + field):
                                if field in fields:
                                    raise ValueError("Duplicate recap publication/modification field")
                                fields[field] = line["text"].split(": ", 1)[-1]
                identity = {"game_id": gid, "source_id": source["source_id"]}
                checked.append({**identity, **fields})
                for field in ("published", "lastModified"):
                    try:
                        value = stamp(fields[field])
                    except (KeyError, ValueError, TypeError, AttributeError):
                        missing.append({**identity, "field": field})
                        continue
                    timestamps.append(value)
                    if value > boundary:
                        late.append({**identity, "field": field, "value": fields[field]})
        results.append({"fold": number, "first_prediction_date": first, "first_prediction_cutoff": boundary.isoformat(),
                        "training_games": len(fitting), "recap_sources_checked": len(checked), "checked_sources": checked,
                        "missing_date_fields": missing, "post_boundary_recap_fields": late,
                        "latest_recap_time": max(timestamps).isoformat() if timestamps else None})
    missing_records = sorted(train.keys() - records.keys())
    return {"status": "pass" if not missing_records and not any(row["missing_date_fields"] or row["post_boundary_recap_fields"] for row in results) else "failed",
            "folds": results, "missing_postgame_game_ids": missing_records,
            "limitation": "Publication/modification metadata audit, not archive proof. Retrieval in 2026 does not prove historical byte availability. Final-score and boxscore correction times remain unproven; raw summaries never become prediction inputs."}


