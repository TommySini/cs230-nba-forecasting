# Cached whole-packet source volume — 2026-11-03

A one-pass researcher reads **1,980 games**, **2,127 source documents**, **233,521,653 characters**, and **105,983,157 Qwen2.5 source tokens**. These are exact local Qwen tokenizer counts, **not GPT-4.1 billed tokens**. No model/API call, download, raw ESPN input, or outcomes read was performed.

| Seasonal split | Games | Source characters | Qwen source tokens | Median/game | P95/game | Max/game | Games >120k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,632 | 189,672,682 | 84,873,639 | 45,738 | 103,227 | 181,564 | 32 |
| dev | 201 | 25,032,316 | 12,025,405 | 73,213 | 110,452 | 239,139 | 10 |
| test | 147 | 18,816,655 | 9,084,113 | 73,920 | 115,788 | 224,690 | 4 |

Overall median **46,876**, P95 **107,697**, maximum **239,139** tokens/game. A hypothetical 120,000-source-token chunk limit needs **2,026 chunks** before overlap and reduction passes.

Selection: exclude 80 new HTML rows explicitly marked modified after cutoff; retain active old sources; check each recorded source timestamp against cutoff; choose the latest eligible timestamp per publisher/game; remove identical raw SHA or exact text within a game. A shared HTML API-container SHA is not a unique article. Timestamp ties prefer the new extraction. The original 424-game snapshot was not changed.

Text: existing new `text_path` contents; old PDF pages use their existing `plain_text` field, with existing `text` as fallback. No truncation, new extraction, source rewriting, or model summarization. All cached pages/sections are included. This is a sizing convention, not reconstruction or content approval.

A sensitivity range of 2–5 characters/token gives **46,704,331–116,760,827** provider input tokens; this is a heuristic range, not a guaranteed bound. Adding 1,000 prompt/source-format tokens per game gives **107,963,157** Qwen-proxy input tokens. A 2,000-token output allowance per game totals **3,960,000** output tokens.

The JSON report records input and tokenizer hashes, selection metadata, every game’s volume, and every selected source path/hash. Input estimates exclude retry costs and do not reproduce the old multiple-stage research workflow. Tables may require layout preservation, and new-source raw/text binding, article identity, eligibility, and readability still require validation. Full seasonal research coverage is not the same as forecast eligibility or a fresh holdout.
