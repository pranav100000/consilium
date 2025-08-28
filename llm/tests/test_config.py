import pytest
import os
from unittest.mock import patch # Import patch

# Patch load_dotenv for all tests in this file to prevent external .env interference
@pytest.fixture(autouse=True)
def mock_load_dotenv(mocker):
    mocker.patch('dotenv.load_dotenv') # Mocks it where config.py would import it
    # We also need to patch it where config.py *calls* it if direct import is used in config.py
    # Assuming config.py does `from dotenv import load_dotenv; load_dotenv()`
    # The above mocker.patch should cover it if config.py is imported after this fixture runs.
    # If config.py does `import dotenv; dotenv.load_dotenv()`, then `mocker.patch('dotenv.load_dotenv')` is correct.
    # Let's also explicitly ensure config is reloaded or patched effectively.
    # For safety, one could also patch 'pydantic_ai_wrapper.config.load_dotenv' if that specific instance is an issue.
    # The simplest is to just ensure config.py uses a mocked load_dotenv.

# Now import the module we are testing AFTER the mock_load_dotenv fixture is defined (due to autouse=True)
from llm import config

@pytest.fixture(autouse=True)
def manage_env_vars(monkeypatch):
    """Clears relevant env vars before each test and restores them after."""
    original_environ = os.environ.copy()
    # Define a list of keys our config module might set or use
    # This is more for cleanup than for controlling the test itself, 
    # as monkeypatch.setenv will handle individual test setup.
    keys_to_clear = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
        "OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OPENAI_BASE_URL",
        "AZURE_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION",
        "GOOGLE-VERTEX_API_KEY", "GCP_PROJECT_ID", "GCP_REGION", "GCP_SERVICE_ACCOUNT_FILE"
    ]
    for key in keys_to_clear:
        if key in os.environ:
            monkeypatch.delenv(key, raising=False)
    
    yield
    
    # Restore original environment state if needed, though pytest usually isolates this.
    # For os.environ, direct modification might persist if not careful, but monkeypatch handles it.
    os.environ.clear()
    os.environ.update(original_environ)


def test_get_api_key_present(monkeypatch):
    monkeypatch.setenv("TESTPROVIDER_API_KEY", "test_key_123")
    assert config.get_api_key("TESTPROVIDER") == "test_key_123"
    monkeypatch.setenv("OPENAI_API_KEY", "openai_specific_key")
    assert config.get_api_key("openai") == "openai_specific_key"

def test_get_api_key_absent():
    # Relies on manage_env_vars to ensure TESTPROVIDER_API_KEY is not set
    assert config.get_api_key("TESTPROVIDER_ABSENT") is None

def test_get_provider_config_basic_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "pk-12345")
    # Explicitly ensure OPENAI_BASE_URL is NOT set for this test
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = config.get_provider_config("openai")
    assert cfg == {"api_key": "pk-12345"}

def test_get_provider_config_openai_key_absent(monkeypatch):
    # Explicitly ensure OPENAI_BASE_URL is NOT set for this test
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    cfg = config.get_provider_config("openai")
    assert cfg == {}

def test_get_provider_config_openai_with_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "pk-xyz")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://custom.openai.url/v1")
    cfg = config.get_provider_config("openai")
    assert cfg == {"api_key": "pk-xyz", "base_url": "http://custom.openai.url/v1"}

def test_get_provider_config_ollama_with_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    cfg = config.get_provider_config("ollama")
    assert cfg == {"base_url": "http://localhost:11434"}

def test_get_provider_config_ollama_with_api_key_and_base_url(monkeypatch):
    # Some OpenAI compatible endpoints used with Ollama might also take a dummy key
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama_dummy_key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11234")
    cfg = config.get_provider_config("ollama")
    assert cfg == {
        "api_key": "ollama_dummy_key",
        "base_url": "http://localhost:11234"
    }

def test_get_provider_config_azure(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "azure-key-value")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
    cfg = config.get_provider_config("azure")
    assert cfg == {
        "api_key": "azure-key-value",
        "azure_endpoint": "https://myaccount.openai.azure.com/",
        "api_version": "2023-05-15"
    }

def test_get_provider_config_google_vertex(monkeypatch):
    # Note: The config.py uses provider_name.upper() for API_KEY, so GOOGLE-VERTEX_API_KEY
    monkeypatch.setenv("GOOGLE-VERTEX_API_KEY", "vertex-ai-key") 
    monkeypatch.setenv("GCP_PROJECT_ID", "my-gcp-project")
    monkeypatch.setenv("GCP_REGION", "us-central1")
    monkeypatch.setenv("GCP_SERVICE_ACCOUNT_FILE", "/path/to/service.json")
    
    cfg = config.get_provider_config("google-vertex")
    assert cfg == {
        "api_key": "vertex-ai-key",
        "project_id": "my-gcp-project",
        "region": "us-central1",
        "service_account_file": "/path/to/service.json"
    }

def test_get_provider_config_partial_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myaccount.openai.azure.com/")
    cfg = config.get_provider_config("azure")
    assert cfg == {
        "azure_endpoint": "https://myaccount.openai.azure.com/"
    }
    assert "api_key" not in cfg

def test_get_provider_config_unknown_provider():
    cfg = config.get_provider_config("unknown_provider_xyz")
    assert cfg == {} # Should return empty dict, not raise error

# To test the dotenv loading itself is more complex as it happens on import.
# One way is to use a subprocess or carefully reload the module.
# However, we are testing the functions that *use* the loaded env vars,
# so if os.getenv works (which it does), and our functions use it, 
# and dotenv sets os.environ (which it does), then the chain is implicitly tested by the above.
# The key is that load_dotenv() in config.py doesn't overwrite monkeypatched vars 
# if monkeypatch is applied correctly around the import or os.environ access.
# The current setup (import config, then monkeypatch in tests) is standard. 