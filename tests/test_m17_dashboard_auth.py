
import pytest
from fastapi.testclient import TestClient
from api.dashboard_server import app
from config.settings import load_settings

client = TestClient(app)
settings = load_settings()

class TestDashboardAuth:
    def test_unauthorized_access(self):
        """Verify 403 when no key provided."""
        response = client.get("/api/health")
        assert response.status_code == 403

    def test_authorized_access(self):
        """Verify 200 when correct key provided."""
        headers = {"X-API-Key": settings.dashboard_api_key}
        response = client.get("/api/health", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_websocket_auth(self):
        """Verify WebSocket connection auth."""
        # Fail without token
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/trading-stream") as websocket:
                pass
        
        # Success with token
        with client.websocket_connect(f"/ws/trading-stream?token={settings.dashboard_api_key}") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "metrics"

if __name__ == "__main__":
    pytest.main([__file__])
