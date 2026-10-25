
#!/usr/bin/env bash
set -euo pipefail

test_db="$(
uv run --extra dev python - <<'PY'
import re

from wikirag.config import load_settings

settings = load_settings()
name = settings.db_name_test

# Validate SQL identifiers before using them in administrative SQL.
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*_test", name):
    raise SystemExit("Unsafe DB_NAME_TEST: use letters, digits and underscores")

print(name)
PY
)"

# Start MariaDB and wait for its health check to pass.
docker compose up -d --wait db

docker compose exec -T -e WIKIRAG_TEST_DB="$test_db" db sh -s <<'CONTAINER'
set -eu

# Account names are included in SQL, so reject unsafe characters.
case "$MARIADB_USER" in
    ""|*[!a-zA-Z0-9_]*)
        echo "Unsafe MARIADB_USER value" >&2
        exit 1
        ;;
esac

# Create the test database without changing the main database.
printf 'CREATE DATABASE IF NOT EXISTS `%s`;\n' "$WIKIRAG_TEST_DB" \
    | MYSQL_PWD="$MARIADB_ROOT_PASSWORD" mariadb -u root

# Allow the existing application user to use the test database.
printf "GRANT ALL PRIVILEGES ON \`%s\`.* TO '%s'@'%%';\n" \
    "$WIKIRAG_TEST_DB" "$MARIADB_USER" \
    | MYSQL_PWD="$MARIADB_ROOT_PASSWORD" mariadb -u root

# Verify that the application user can connect successfully.
MYSQL_PWD="$MARIADB_PASSWORD" mariadb -u "$MARIADB_USER" \
    "$WIKIRAG_TEST_DB" -e 'SELECT DATABASE() AS test_database;'
CONTAINER
