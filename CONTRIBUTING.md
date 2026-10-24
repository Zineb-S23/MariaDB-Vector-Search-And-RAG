# Contributing to WikiRAG

**Project:** MariaDB Area 4 — Vector Search and RAG, Natively
**Team:** Three collaborating contributors; no designated team lead
**Source of truth:** The official hackathon brief governs deliverables. [`docs/interfaces.md`](docs/interfaces.md) governs proposed internal APIs once all contributors approve it.

## 1. Working agreement

We use short-lived feature branches and reviewed GitHub pull requests (PRs). **Never commit directly to `main`.** The evaluator reads `main`, so tested, accepted work must be merged there regularly—not left only on feature branches.

- `main` must remain runnable and reproducible.
- Each PR should have one clear purpose; aim for roughly 400 changed lines or fewer when practical. Split large work into independently testable PRs.
- Make small, meaningful commits after tests pass. A commit saves work locally; a push uploads it; only a merged PR puts it on `main`.
- Use normal merge commits (GitHub **Create a merge commit**) rather than squash merges, to preserve meaningful commit history, as recommended in the architecture blueprint.
- Do not rebase or force-push a shared branch.
- Do not commit `.env`, credentials, API keys, database passwords, virtual environments, generated model weights, or large unapproved data files.
- Everyone participates in reviewing shared contracts and integration decisions. **Folder ownership is not a management hierarchy.**

## 2. Areas of responsibility

| Contributor | Primary area | Typical files |
| --- | --- | --- |
| Database/integration contributor | MariaDB schema, connections, repository, vector/hybrid SQL, real-DB tests, integration | `src/wikirag/db/`, `sql/`, `tests/integration/` |
| Zineb | Wikipedia collection, snapshots, cleaning/chunking, embeddings, ingestion | `src/wikirag/embeddings.py`, `src/wikirag/ingestion/`, `data/` |
| Marwan | Retrieval, RAG, evaluation, experiments | `src/wikirag/retrieval/`, `src/wikirag/rag/`, `src/wikirag/evaluation/`, evaluation scripts/results |
| Everyone | Code review, API compatibility, README accuracy, integration and reproducibility | `src/wikirag/models.py`, `src/wikirag/config.py`, `src/wikirag/errors.py`, `docs/interfaces.md`, `README.md` |

Before editing files outside your primary area, coordinate with the relevant contributor. For shared interfaces, see Section 8.

## 3. Branch naming and planned work

Branch from the **latest `main`**, using `feature/<topic>` or `docs/<topic>`.

| Branch | Main work |
| --- | --- |
| `feature/foundation` | Project setup, shared models/config/errors, test fakes, interfaces, documentation and PR template |
| `feature/database-layer` | SQL schema, MariaDB repository, hybrid vector search, database tests |
| `feature/wikipedia-ingestion` | Seed titles, Wikipedia snapshot, cleaning and chunking |
| `feature/embedding-pipeline` | Sentence-transformer embeddings and ingestion pipeline |
| `feature/retrieval` | Semantic/hybrid retriever and associated tests |
| `feature/rag-pipeline` | Context building, generator, RAG pipeline |
| `feature/evaluation-metrics` | Ranking metrics and judged queries |
| `feature/evaluation-runner` | Evaluation execution and results |
| `feature/ci-and-docker` | CI checks and remaining Docker/reproducibility improvements |
| `feature/cli` | Command-line interface |
| `feature/final-integration` | Fresh-clone verification, README and integration fixes |

These are planning names from the architecture blueprint, not proof that branches already exist. Open and merge finished PRs as early as possible before the **Thursday, 29 October 2026** evaluation.

## 4. Starting a new branch

Run from the repository root in WSL Ubuntu:

```bash
# Make sure your existing work is committed or safely set aside first.
git status --short

git switch main
git pull --ff-only origin main
git switch -c feature/<topic>
```

Replace `<topic>` with a meaningful name; do **not** type the angle brackets literally.

For the **existing** foundation branch, stay on `feature/foundation` until its PR is merged. Do not create another foundation branch.

## 5. While implementing

### One tested checkpoint at a time

1. Implement a small, coherent piece.
2. Update relevant tests and documentation in the same piece of work.
3. Run the relevant tests and lint checks.
4. Review the diff for accidental changes and secrets.
5. Commit with a meaningful message.
6. Push when you want a backup or are ready for collaboration.

Commands:

```bash
python -m ruff format .
make lint
python -m pytest -q
git diff --check
git status --short
git diff
```

For a database-dependent change, also run the applicable integration tests against the project's **test database**, not a production database. Record the actual test command and result in the PR. If a full suite depends on unfinished components, state that clearly rather than claiming everything passed.

### Example commit messages

```text
build: add packaging and compose configuration
feat: add shared models, errors and config
feat: add in-memory repository and fake embedder
test: cover hybrid search filters and ingestion rollback
feat: add transactional article ingestion
fix: avoid duplicate rows in category-filtered search
docs: document shared interfaces and contribution workflow
```

Use `feat`, `fix`, `test`, `docs`, `refactor`, or `build` (or a clear plain-English message). Avoid the `chore:` prefix.

### Committing versus pushing

```bash
# Stage only the files relevant to this checkpoint.
git add path/to/file.py tests/unit/test_related.py

git diff --cached --check
git diff --cached --name-only

git commit -m "feat: describe the tested change"
git push -u origin "$(git branch --show-current)"
```

**Commit** updates local Git history. **Push** updates the branch on GitHub. **Merge** puts an approved branch into `main`.

## 6. Opening and reviewing a pull request

1. Push the feature branch to GitHub.
2. Open **Pull requests → New pull request**.
3. Set **base = `main`** and **compare = your feature branch**.
4. Use the PR template to explain the work, tests, interface changes, documentation, and any known limitations.
5. Request at least one review from another contributor; request reviews from **all affected module owners** when public contracts change.
6. Address review comments and rerun checks.
7. Merge using **Create a merge commit** once the agreed checks and reviews pass.
8. Delete the merged feature branch on GitHub.

The **foundation PR** needs review by both Zineb and Marwan because their components will depend on its models, fakes, and interfaces. Each contributor should be able to install dependencies and run the foundation tests before approval.

For other PRs, the reviewer should ideally own a component that **consumes** the proposed interface. Do not approve your own PR as its sole reviewer.

If your repository permissions allow it, configure `main` branch protection: require PRs, at least one review approval, and passing checks when CI is ready; disallow force pushes. Do not delay merging tested work just because GitHub settings are unavailable.

## 7. Staying up to date and resolving conflicts

Every working day—and before requesting PR review—bring the latest `main` into your **feature branch**:

```bash
git switch feature/<your-branch>
git fetch origin
git merge origin/main
```

Replace `feature/<your-branch>` with the actual branch name.

If Git reports conflicts:

1. Open the conflicted files and inspect both changes carefully.
2. Consult the primary contributor before resolving a conflict in their area.
3. Do not discard someone else's changes just to make Git stop complaining.
4. Resolve the conflict markers, stage the resolved files, and complete the merge commit.
5. Run lint and relevant tests again, then push.

Never use `git push --force` to fix integration conflicts. After another PR merges to `main`, update your open branch promptly.

## 8. Shared interface change protocol

[`docs/interfaces.md`](docs/interfaces.md) and the definitions in `src/wikirag/models.py` establish the agreed contract for cross-module data.

Before changing a public function name, parameter, default, return type, exception, model field, embedding dimension, or search semantics:

1. Describe the proposed change and why the current contract cannot work.
2. Identify the producer module and **all consuming modules**.
3. Obtain agreement from affected contributors; nobody unilaterally changes a shared contract.
4. Update the implementation, `docs/interfaces.md`, `tests/fakes.py` where applicable, and contract tests together in a reviewed PR.
5. Verify the fake and real repository behave consistently where applicable.

AI coding assistants must not silently rename agreed public interfaces or modify shared dataclasses.

## 9. Testing expectations

| Change | Minimum verification |
| --- | --- |
| Pure Python models, helpers or parsing | Ruff and focused unit tests |
| Fake embedder or fake repository | Fake behavior and contract tests |
| SQL schema, repository, or transactions | Real MariaDB integration tests plus relevant unit tests |
| Wikipedia ingestion | Parsing/chunking tests and repeat-ingestion/idempotency test |
| Retrieval | Fake-repository tests and real-database search tests when available |
| RAG | Context-citation tests and the no-hit/no-generation guard |
| Evaluation | Hand-checked ranking metrics and reproducible, recorded runs |
| Final integration | Full tests, end-to-end run, and fresh-clone setup verification |

Use the same shared dataclasses and function signatures that the other contributors import. A green test for one module is not evidence that all modules integrate successfully.

## 10. Documentation and reproducibility

Update documentation as part of the corresponding feature PR, not at the end of the project:

- **`README.md`:** exact prerequisites, setup, Docker/MariaDB version, setup/run commands, corpus source and reproduction steps.
- **`docs/interfaces.md`:** public APIs, shared models, filters, errors, behavior and approved changes.
- **Database docs:** SQL schema rationale, `VECTOR(384)`, HNSW index settings, transactional design, cosine and hybrid SQL examples.
- **Ingestion docs:** Wikipedia source, snapshot format, cleaning, chunk size/overlap, embedding model and dimensions.
- **Retrieval/evaluation docs:** semantic and hybrid modes, judged question set, recall/precision/MRR, latency methods and measured results.
- **Results:** include settings and version details; never fabricate numbers or report unrun experiments as successful.

Before the evaluation, verify that everything needed to reproduce the project is merged into **`main`**, and that a teammate can follow the README from a fresh clone.

## 11. Secrets and generated files

- Put local credentials in `.env`, not `.env.example` or Markdown documentation.
- Keep `.env`, `.venv`, caches, generated model checkpoints, and unwanted database/data exports out of Git.
- Never paste real passwords, tokens, or API keys into PRs, issues, screenshots, or logs.
- Inspect `git diff --cached` before committing, especially when using `git add .`.
- Use MariaDB's designated test database for tests that modify tables or data.

## 12. Before merging: PR checklist

- [ ] The PR addresses one coherent deliverable.
- [ ] The current branch includes the latest necessary changes from `main`.
- [ ] Ruff and relevant tests pass; any unrun tests are explicitly stated.
- [ ] Function signatures and shared models agree with `docs/interfaces.md`.
- [ ] No secrets, local paths, virtual environments, or unwanted generated files are staged.
- [ ] Documentation, schema notes, and example commands are updated as needed.
- [ ] A contributor familiar with the affected interface has reviewed the changes.
- [ ] The feature is ready to be merged into `main` without breaking the runnable project.

**Priority:** Finish and merge useful, verified work early. An excellent implementation left only on a feature branch will not be visible to an evaluator who checks `main`.
