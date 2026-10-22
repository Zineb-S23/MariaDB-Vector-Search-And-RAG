
# Convenient development commands for the MariaDB WikiRAG project.

.PHONY: up down ps version test lint

# Start MariaDB using Docker Compose.
up:
	docker compose up -d

# Stop containers without deleting the database volume.
down:
	docker compose down

# Display container status.
ps:
	docker compose ps

# Display the running MariaDB server version.
version:
	docker compose exec db mariadb -uroot -p -e "SELECT VERSION();"

# Run the Python test suite.
test:
	python -m pytest

# Check Python code quality.
lint:
	python -m ruff check .
