# Teacher API setup

**Execution-platform update (2026-11-07):** the user now requires all experiment
compute on AWS and requested architecture review before execution. Use
[the AWS architecture](bam_aws_architecture.md) for new work: managed teacher
calls through Bedrock with AWS authorization, research and training on EC2,
and artifacts on S3, within the current $200 cap. The direct-provider/Modal
instructions below describe the previous plan and must not launch the new run.

An API model would generate and check TRAIN postmortems/gold targets. The
researcher and forecaster would still be fine-tuned on Modal. No API provider has
been selected or called yet, and the remaining $150 ceiling has not increased.

1. Create an API account with [OpenAI](https://platform.openai.com/) or
   [Anthropic](https://platform.claude.com/). Check existing subscription credits
   before buying credit: eligible [ChatGPT Pro/Business plans](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)
   and [Claude Max/Ultra/Team plans](https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers)
   currently offer a linking flow. Eligibility and balances must be checked in
   the account; they are not assumed for this project.
2. Create a project API key in that provider's dashboard.
3. In this project's terminal, run one of the following. Paste the key at the
   hidden prompt, not into chat or a source file.

```sh
.venv/bin/python scripts/configure_bam_api_key.py --provider openai
```

Or, for Anthropic:

```sh
.venv/bin/python scripts/configure_bam_api_key.py --provider anthropic
```

The helper stores the key under the git-ignored
`data/processed/bam_api_credentials/` folder, with owner-only file and directory
permissions. It makes no network request and causes no charge. Running it again
replaces the saved key for that provider. Its `--check` option reports only file
presence; this does not verify API access or billing.

The first proposed API work is the existing, frozen 16-claim TRAIN diagnostic,
followed by the four-game complete source/report audit. A teacher must pass the
source audit before its reports become training targets. Modern API teachers
stay outside the held-out researcher/critic path. The API boundary and shared
token/GPU cost ledger are prepared; provider execution is not yet integrated.

The earlier $35 Sonnet/Sol or $70 Opus illustration assumes 15 million input and
4 million output tokens at current batch prices. It covers that hypothetical
teacher stage only, excluding additional review/repair, source research, student
training/generation and forecasting. It is not a full-experiment quote. See
[OpenAI pricing](https://developers.openai.com/api/docs/pricing) and
[Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing).
