import os
from dotenv import load_dotenv

# Load environment variables from .env file at the root of the project
# Assumes .env is in the same directory as the script that imports this,
# or in the current working directory. For a more robust path, especially
# if pydantic_ai_wrapper is a package, you might need to adjust.
# For now, let's assume the .env is at the project root where you run your main script.
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
else:
    # Fallback for cases where .env might be in the current working directory
    # or if the above path logic doesn't find it.
    # python-dotenv will also search common locations by default if dotenv_path is None.
    load_dotenv() 

def get_api_key(provider_name: str) -> str | None:
    """
    Retrieves the API key for a given provider from environment variables.
    Expects environment variables in the format: PROVIDER_NAME_API_KEY
    e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY
    """
    key_name = f"{provider_name.upper()}_API_KEY"
    return os.getenv(key_name)

def get_provider_config(provider_name: str) -> dict:
    """
    Retrieves specific configurations for a provider.
    For now, it primarily focuses on API keys.
    This can be expanded to include other provider-specific settings
    if they are also stored in environment variables or a config file.
    """
    config = {}
    api_key = get_api_key(provider_name)
    if api_key:
        config['api_key'] = api_key

    # Example for provider-specific base URL (e.g., for Ollama or OpenAI-compatible)
    base_url_key = f"{provider_name.upper()}_BASE_URL"
    base_url = os.getenv(base_url_key)
    if base_url:
        config['base_url'] = base_url
        
    # Example for Azure specific configs
    if provider_name.lower() == 'azure':
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT') # Or a more specific name
        azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        if azure_endpoint:
            config['azure_endpoint'] = azure_endpoint
        if azure_api_version:
            config['api_version'] = azure_api_version

    # Example for Gemini Vertex AI specific configs
    if provider_name.lower() == 'google-vertex':
        gcp_project_id = os.getenv('GCP_PROJECT_ID')
        gcp_region = os.getenv('GCP_REGION')
        gcp_service_account_file = os.getenv('GCP_SERVICE_ACCOUNT_FILE') # Path to JSON
        if gcp_project_id:
            config['project_id'] = gcp_project_id
        if gcp_region:
            config['region'] = gcp_region
        if gcp_service_account_file:
            config['service_account_file'] = gcp_service_account_file
            
    return config

# You can also pre-load known keys if preferred:
# OPENAI_API_KEY = get_api_key("OPENAI")
# ANTHROPIC_API_KEY = get_api_key("ANTHROPIC")
# GEMINI_API_KEY = get_api_key("GEMINI") # For Generative Language API 