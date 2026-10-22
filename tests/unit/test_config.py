"""
Unit tests for the shared WikiRAG configuration module.

These tests verify environment variable loading, configuration validation,
database safety, default settings, and credential protection.

The tests use temporary environment variables and never connect to MariaDB.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import pytest

from wikirag import config
from wikirag.errors import ValidationError


@pytest.fixture
def isolated_config_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Provide a predictable environment for configuration tests.

    The real .env file is not loaded, preventing local credentials
    or machine-specific settings from affecting test results.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to
            temporarily modify environment variables and functions.

    Returns:
        None: The environment is prepared for the requesting test.
    """

    # Prevent tests from reading the developer's private .env file.
    monkeypatch.setattr(
        config,
        "load_dotenv",
        Mock(return_value=False),
    )

    variable_names = (
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
        "DB_NAME_TEST",
        "EMBEDDING_MODEL",
        "LOG_LEVEL",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "LLM_BASE_URL",
    )

    # Remove existing values to avoid machine-specific test results.
    for name in variable_names:
        monkeypatch.delenv(name, raising=False)

    # Set safe, fake credentials for the test environment.
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    monkeypatch.setenv("DB_PORT", "13307")
    monkeypatch.setenv("DB_NAME", "wikirag")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "fake_test_password")


def test_load_settings_with_valid_environment(
    isolated_config_environment: None,
) -> None:
    """
    Verify that valid environment variables produce correct settings.

    Args:
        isolated_config_environment (None): Fixture providing isolated
            environment variables with safe test credentials.

    Returns:
        None: The test passes when all settings match expectations.

    Raises:
        AssertionError: If any loaded configuration value is incorrect.
    """

    settings = config.load_settings()

    assert settings.db.host == "127.0.0.1"
    assert settings.db.port == 13307
    assert settings.db.database == "wikirag"
    assert settings.db.user == "test_user"
    assert settings.db.password == "fake_test_password"


def test_default_configuration_values(
    isolated_config_environment: None,
) -> None:
    """
    Verify defaults for optional application settings.

    Args:
        isolated_config_environment (None): Isolated environment fixture.

    Returns:
        None: The test passes when optional settings have correct defaults.

    Raises:
        AssertionError: If any default configuration value changes.
    """

    settings = config.load_settings()

    assert settings.db_name_test == "wikirag_test"
    assert settings.embedding_model == config.EMBEDDING_MODEL_NAME
    assert config.EMBEDDING_DIM == 384
    assert settings.log_level == "INFO"

    assert settings.llm_provider == "none"
    assert settings.llm_model is None
    assert settings.llm_api_key is None
    assert settings.llm_base_url is None


@pytest.mark.parametrize(
    "missing_variable",
    [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ],
)
def test_missing_required_environment_variable(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    missing_variable: str,
) -> None:
    """
    Verify that missing database configuration is rejected.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.
        missing_variable (str): Required variable to remove.

    Returns:
        None: The test passes when ValidationError is raised.

    Raises:
        AssertionError: If missing credentials are accepted.
    """

    monkeypatch.delenv(missing_variable)

    with pytest.raises(ValidationError, match=missing_variable):
        config.load_settings()


@pytest.mark.parametrize(
    "variable_name",
    [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ],
)
def test_blank_required_environment_variable(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    variable_name: str,
) -> None:
    """
    Verify that whitespace-only required settings are rejected.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.
        variable_name (str): Required variable to replace with whitespace.

    Returns:
        None: The test passes when ValidationError is raised.

    Raises:
        AssertionError: If an empty required setting is accepted.
    """

    monkeypatch.setenv(variable_name, "   ")

    with pytest.raises(ValidationError, match=variable_name):
        config.load_settings()


@pytest.mark.parametrize(
    "invalid_port",
    [
        "not_a_number",
        "3307.5",
        "",
        "0",
        "-1",
        "65536",
    ],
)
def test_invalid_database_port(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    invalid_port: str,
) -> None:
    """
    Verify that invalid MariaDB port values are rejected.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.
        invalid_port (str): Invalid port value to test.

    Returns:
        None: The test passes when ValidationError is raised.

    Raises:
        AssertionError: If an invalid port is accepted.
    """

    monkeypatch.setenv("DB_PORT", invalid_port)

    with pytest.raises(ValidationError, match="DB_PORT"):
        config.load_settings()


def test_default_database_port(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify the default Docker development port.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for removing DB_PORT.

    Returns:
        None: The test passes when the default port is 13307.

    Raises:
        AssertionError: If the default port is incorrect.
    """

    monkeypatch.delenv("DB_PORT")

    settings = config.load_settings()

    assert settings.db.port == 13307


@pytest.mark.parametrize(
    "test_database_name",
    [
        "wikirag",
        "production",
        "wikirag_testing",
        "",
    ],
)
def test_invalid_test_database_name(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
    test_database_name: str,
) -> None:
    """
    Verify that unsafe integration-test database names are rejected.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.
        test_database_name (str): Invalid test database name.

    Returns:
        None: The test passes when ValidationError is raised.

    Raises:
        AssertionError: If unsafe test database naming is accepted.
    """

    monkeypatch.setenv("DB_NAME_TEST", test_database_name)

    with pytest.raises(ValidationError, match="DB_NAME_TEST"):
        config.load_settings()


def test_test_database_must_differ_from_main_database(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that test and application databases cannot share a name.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.

    Returns:
        None: The test passes when matching names are rejected.

    Raises:
        AssertionError: If matching database names are accepted.
    """

    monkeypatch.setenv("DB_NAME", "wikirag_test")
    monkeypatch.setenv("DB_NAME_TEST", "wikirag_test")

    with pytest.raises(ValidationError, match="DB_NAME_TEST"):
        config.load_settings()


def test_empty_embedding_model(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that a blank embedding model identifier is rejected.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.

    Returns:
        None: The test passes when ValidationError is raised.

    Raises:
        AssertionError: If a blank embedding model is accepted.
    """

    monkeypatch.setenv("EMBEDDING_MODEL", "   ")

    with pytest.raises(ValidationError, match="EMBEDDING_MODEL"):
        config.load_settings()


def test_invalid_logging_level(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that unsupported logging levels are rejected.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for modifying variables.

    Returns:
        None: The test passes when ValidationError is raised.

    Raises:
        AssertionError: If an unsupported logging level is accepted.
    """

    monkeypatch.setenv("LOG_LEVEL", "VERY_VERBOSE")

    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        config.load_settings()


def test_optional_configuration_overrides(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that optional settings can be overridden by environment variables.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for setting overrides.

    Returns:
        None: The test passes when overrides are loaded correctly.

    Raises:
        AssertionError: If optional settings ignore environment values.
    """

    monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    monkeypatch.setenv("LLM_PROVIDER", "example_provider")
    monkeypatch.setenv("LLM_MODEL", "example-model")
    monkeypatch.setenv("LLM_API_KEY", "fake_api_key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/api")

    settings = config.load_settings()

    assert settings.embedding_model == "custom-model"
    assert settings.log_level == "WARNING"
    assert settings.llm_provider == "example_provider"
    assert settings.llm_model == "example-model"
    assert settings.llm_api_key == "fake_api_key"
    assert settings.llm_base_url == "https://example.com/api"


def test_credentials_are_hidden_from_representation(
    isolated_config_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that settings representations do not expose credentials.

    Args:
        isolated_config_environment (None): Isolated environment fixture.
        monkeypatch (pytest.MonkeyPatch): Fixture for setting a fake API key.

    Returns:
        None: The test passes when credentials are absent from repr().

    Raises:
        AssertionError: If a credential appears in an object representation.
    """

    monkeypatch.setenv("LLM_API_KEY", "fake_secret_api_key")

    settings = config.load_settings()
    representation = repr(settings)

    # Both database passwords and optional API keys must be hidden.
    assert "fake_test_password" not in representation
    assert "fake_secret_api_key" not in representation


def test_settings_are_immutable(
    isolated_config_environment: None,
) -> None:
    """
    Verify that loaded configuration fields cannot be reassigned.

    Args:
        isolated_config_environment (None): Isolated environment fixture.

    Returns:
        None: The test passes when reassignment is rejected.

    Raises:
        AssertionError: If reassignment does not raise FrozenInstanceError.
    """

    settings = config.load_settings()

    with pytest.raises(FrozenInstanceError):
        settings.log_level = "DEBUG"

    with pytest.raises(FrozenInstanceError):
        settings.db.port = 3306


def test_dotenv_loading_configuration(
    isolated_config_environment: None,
) -> None:
    """
    Verify that the loader uses the project .env without overriding OS values.

    Args:
        isolated_config_environment (None): Isolated environment fixture.

    Returns:
        None: The test passes when load_dotenv receives the expected arguments.

    Raises:
        AssertionError: If the .env path or precedence policy changes.
    """

    config.load_settings()

    expected_root = Path(config.__file__).resolve().parents[2]
    expected_path = expected_root / ".env"

    # The fixture replaces load_dotenv with a mock, so no secret file is read.
    config.load_dotenv.assert_called_once_with(
        dotenv_path=expected_path,
        override=False,
    )
