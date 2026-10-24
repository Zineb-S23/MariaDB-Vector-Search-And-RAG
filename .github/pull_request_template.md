# Pull request summary

## What changed?

<!-- Explain what this PR adds, fixes, or changes in 2–5 sentences. -->

## Why is this change needed?

<!-- Link an issue, blueprint section, or requirement when relevant. -->

## Component(s) affected

- [ ] Shared contracts / configuration
- [ ] Wikipedia ingestion / cleaning / chunking
- [ ] Embeddings
- [ ] MariaDB schema / database repository
- [ ] Retrieval / hybrid SQL search
- [ ] RAG / context / generation
- [ ] Evaluation / experiments
- [ ] Documentation / developer tooling

## Interfaces and integration

- [ ] Public names, argument types, return types, and model fields match `docs/interfaces.md` and `src/wikirag/models.py`.
- [ ] Cross-component imports and integration points have been checked.
- [ ] Any interface change is explained below and reviewed by affected contributors.
- [ ] If the repository or embedding contract changed, matching fakes and contract tests were updated.
- [ ] If this PR does not affect interfaces, I marked these items as not applicable below.

**Interface changes or N/A:**

<!-- List exact before/after signatures, consumers, and approvals. -->

## Testing evidence

- [ ] Relevant unit tests pass.
- [ ] Relevant integration tests pass against MariaDB 11.8, or the reason they were not run is stated below.
- [ ] `make lint` passes.
- [ ] `python -m pytest -q` passes, or known failures are documented below.
- [ ] Added/changed behavior is covered by tests that can fail meaningfully.

**Commands run and actual results:**

```text
# Paste commands and short, real output here.
```

**Tests not run / known failures / N/A:**

<!-- Be explicit; do not tick tests that were not run. -->

## Database and retrieval checks (when applicable)

- [ ] MariaDB version and schema/index changes are documented.
- [ ] Vector model/dimension and cosine-distance behavior remain consistent.
- [ ] Hybrid filters and joins preserve the agreed semantics.
- [ ] Writes are atomic; update, skip, and rollback behavior was tested.
- [ ] Search ordering, `top_k`, and no-match behavior were checked.
- [ ] Not applicable; explain below.

**Notes or N/A:**

## Documentation and reproducibility

- [ ] README, setup instructions, and related documentation were updated where necessary.
- [ ] New environment variables are documented in `.env.example` using placeholders only.
- [ ] No `.env`, credentials, tokens, personal data, or machine-specific paths were committed.
- [ ] Any measured retrieval quality or latency values are backed by actual reproducible runs.
- [ ] Commands, data sources, and relevant model/index settings are recorded where applicable.

## Git and review

- [ ] This branch is up to date with `origin/main`, or the difference is explained below.
- [ ] The diff includes only intentional changes; unrelated generated files are excluded.
- [ ] At least one other teammate has reviewed the changes before merge.
- [ ] Any affected contributor has approved changes to shared interfaces.
- [ ] Required checks pass before merging into `main`.

**Reviewer(s) / outstanding questions:**

<!-- Request review from another teammate; there is no special team-lead approval role. -->

## How to verify manually (optional)

```bash
# Minimal reproduction or smoke-test steps, if applicable.
```

## Screenshots or sample output (optional)

<!-- Include only if this PR changes visible output or a CLI workflow. -->
