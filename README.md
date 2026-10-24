A project for area 4, "Vector search and RAG, natively" for the [MariaDB student database projects, 2026-09](https://mariadb.org/bachelor_hackathon_2026-09/).

# MariaDB WikiRAG

## 1. Project Overview

WikiRAG is a Wikipedia retrieval and retrieval-augmented generation (RAG) project built using **MariaDB 11.8** and its native vector search capabilities.

The project aims to combine:

- Semantic search using vector embeddings.
- Relational SQL filtering and joins.
- Native MariaDB vector indexing.
- Transactionally consistent storage.
- Retrieval-augmented generation.
- Evaluation using recall and latency measurements.

Unlike an architecture that requires a separate vector database, WikiRAG aims to store article metadata and embeddings together in MariaDB.

**Current status:** The project foundation is implemented. The database repository, Wikipedia ingestion, retrieval, RAG, and evaluation components are under development.

---

## 2. Technology Stack

| Component | Technology |
|-----------|------------|
| Programming language | Python |
| Database | MariaDB 11.8 |
| Vector storage | MariaDB native VECTOR |
| Vector indexing | HNSW |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding dimensions | 384 |
| Dependency management | uv |
| Database environment | Docker Compose |
| Unit testing | pytest |
| Code quality | Ruff |
| Version control | Git and GitHub |

---

## 3. Prerequisites

Before installing the project, ensure you have:

- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker and Docker Compose
- Internet access for the initial dependency and embedding-model downloads

The foundation and CPU-only embedding installation have been tested using **Python 3.14.4 on WSL Ubuntu**.

The project's supported Python versions are declared in `pyproject.toml`.

---

## 4. Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/melsayed-eg/MariaDB-Vector-Search-And-RAG.git

cd MariaDB-Vector-Search-And-RAG
```

### Step 2: Install Project Dependencies

Run:

```bash
uv sync --extra dev --extra embeddings --locked
```

This command:

1. Creates or updates the project's `.venv` virtual environment.
2. Installs the required Python dependencies.
3. Installs development tools such as pytest and Ruff.
4. Installs Sentence Transformers and CPU-only PyTorch.
5. Uses the dependency versions recorded in `uv.lock`.

**An NVIDIA GPU is not required.**

The CPU-only PyTorch package source is configured in `pyproject.toml`, so contributors do not need to install PyTorch separately.

### Step 3: Activate the Virtual Environment

On Linux or WSL Ubuntu:

```bash
source .venv/bin/activate
```

Alternatively, use `uv run` to execute project commands without manually activating the virtual environment.

---

## 5. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and replace the example passwords with your own local credentials.

The configuration includes:

- MariaDB connection settings.
- Application database name.
- Test database name.
- Embedding model.
- Logging settings.
- Optional LLM configuration.

**Security:** Never commit `.env`, passwords, API keys, or other credentials to GitHub.

---

## 6. Starting MariaDB

The project uses MariaDB 11.8 through Docker Compose.

Start the database:

```bash
docker compose up -d
```

Check the running containers:

```bash
docker compose ps
```

Stop the database:

```bash
docker compose down
```

The default development configuration exposes MariaDB on:

```text
Host: 127.0.0.1
Port: 13307
```

Database schema initialization, test-database setup, and Wikipedia data loading will be documented as their implementations are completed.

---

## 7. Running Tests

### Run Unit Tests

```bash
uv run --extra dev --extra embeddings python -m pytest -m unit -q
```

The current foundation has **104 passing unit tests**.

These tests cover the implemented shared foundation components, including:

- Shared data models.
- Configuration validation.
- Custom exceptions.
- Fake embedding and repository implementations.
- Repository hygiene checks.

### Run Ruff Linting

```bash
uv run --extra dev ruff check .
```

### Check Code Formatting

```bash
uv run --extra dev ruff format --check .
```

Integration and end-to-end tests will be added alongside the corresponding database and application implementations.

---

## 8. Embedding Model

WikiRAG uses the following embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts text into a **384-dimensional numerical vector**.

These vectors will be stored using MariaDB's native `VECTOR(384)` type.

The project's default installation uses CPU-only PyTorch.

### Verify the Embedding Model

After installation, run:

```bash
uv run --extra embeddings python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); vector = model.encode('MariaDB supports vector search.'); print('Embedding dimensions:', vector.shape); assert vector.shape == (384,)"
```

Expected output:

```text
Embedding dimensions: (384,)
```

The initial execution may download the model weights.

---

## 9. Dependency Management

The project uses **uv** for Python dependency and environment management.

Two files control dependency installation:

### `pyproject.toml`

Defines the project's direct dependencies, optional dependency groups, and package-source configuration.

### `uv.lock`

Records resolved dependency versions and package information to support reproducible installation.

The project configures CPU-only PyTorch rather than requiring NVIDIA CUDA libraries.

The previously used `requirements.lock` was an environment-specific package snapshot and has been replaced by the uv-based dependency management approach.

When installing the project, use:

```bash
uv sync --extra dev --extra embeddings --locked
```

Do not manually install individual dependencies unless you are intentionally changing the project's dependency configuration.

---

## 10. Project Architecture

The application is organized into separate components.

| Component | Responsibility |
|-----------|----------------|
| Database | MariaDB connections, schema, transactions, and vector search |
| Ingestion | Wikipedia collection, cleaning, chunking, and embedding |
| Retrieval | Semantic search and relational filtering |
| RAG | Building context and generating grounded responses |
| Evaluation | Measuring retrieval quality and query latency |

The shared interfaces are defined in:

[docs/interfaces.md](docs/interfaces.md)

---

## 11. Development Workflow

Contributors work on feature branches and submit pull requests for review.

Before submitting changes:

1. Run the relevant tests.
2. Run Ruff linting and formatting checks.
3. Update documentation where necessary.
4. Review the Git diff.
5. Commit tested changes.
6. Push the feature branch and open a pull request.

The `main` branch should contain reviewed, runnable work.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the detailed Git workflow and contribution guidelines.

---

## 12. Team Responsibilities

| Area | Responsibility |
|------|----------------|
| Database and integration contributor | MariaDB schema, repository implementation, native vector search, SQL filtering, and integration tests |
| Zineb | Wikipedia collection, text cleaning, chunking, embeddings, and ingestion |
| Marwan | Retrieval, RAG pipeline, evaluation, and experiments |

All contributors participate in reviewing shared interfaces, integration decisions, and pull requests.

---

## 13. Evaluation Plan

The completed project will be evaluated using:

- A ground-truth set of Wikipedia search questions.
- Retrieval recall measurements.
- Query latency measurements.
- Comparison of semantic and SQL-constrained retrieval.
- Reproducible experiments.

The final documentation will explain:

- Chunk size and overlap.
- Embedding model selection.
- HNSW index parameters.
- Database schema design.
- SQL query examples.
- Evaluation methodology and results.

These features are not yet implemented in the current foundation.

---

## 14. Current Development Status

### Implemented

- Project structure and Python packaging.
- Shared models and interfaces.
- Configuration handling.
- Custom exceptions.
- Fake embedding model.
- In-memory repository.
- Foundation unit tests.
- Docker Compose configuration.
- Dependency management and CPU-only embedding installation.

### Under Development

- MariaDB schema and real database repository.
- Native VECTOR and HNSW indexing.
- Wikipedia collection and ingestion.
- Semantic and hybrid SQL retrieval.
- RAG pipeline.
- Recall and latency evaluation.
- End-to-end integration tests.
- Complete reproducibility and final results.

---

## 15. Additional Documentation

- [Shared Interfaces](docs/interfaces.md)
- [Contribution Guidelines](CONTRIBUTING.md)
- [MariaDB Project Assignment](https://mariadb.org/bachelor_hackathon_2026-09/)
- [MariaDB Documentation](https://mariadb.com/docs/)
- [uv Documentation](https://docs.astral.sh/uv/)