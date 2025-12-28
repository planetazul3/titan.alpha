
from fastapi import Header, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from config.settings import load_settings

# Initialize settings
settings = load_settings()
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(
    api_key_header: str = Security(api_key_header),
    token: str = Query(None) # Support query param for WebSocket
):
    """
    Verify API key from header or query parameter.
    """
    if api_key_header == settings.dashboard_api_key.get_secret_value():
        return api_key_header
        
    if token == settings.dashboard_api_key.get_secret_value():
        return token
        
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
    )
