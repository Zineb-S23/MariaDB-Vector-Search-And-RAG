
# Convenient development commands for WikiRAG.

.PHONY: up down ps version test lint test-unit test-db setup-test-db

up:
	docker compose up -d

down:
	docker compose down

ps:
	docker compose ps

version:
	docker compose exec db mariadb -uroot -p -e "SELECT VERSION();"

# Run all tests, including tests against the separate MariaDB test database.
test:
	uv run --extra dev python -m pytest -q

# Run fast tests without requiring MariaDB.
test-unit:
	uv run --extra dev python -m pytest -m unit -q

# Provision the isolated test database and run integration tests.
setup-test-db:
	bash scripts/create-test-db.sh

test-db:
	uv run --extra dev python -m pytest -m integration -q

lint:
	uv run --extra dev ruff check .
	uv run --extra dev ruff format --check .
