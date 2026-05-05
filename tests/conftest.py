import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path so we can import from Lumora_Intelligence
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set mock environment variables before importing app
os.environ["INTERNAL_API_KEY"] = "test-internal-key"
os.environ["APP_ENV"] = "testing"

@pytest.fixture(autouse=True)
def mock_db_reader():
    """Mocks db.reader functions to prevent real DB queries."""
    with patch("services.intelligence_service.get_daily_metrics") as mock_metrics:
        with patch("services.intelligence_service.get_products") as mock_products:
            with patch("services.intelligence_service.get_data_summary") as mock_summary:
                with patch("services.intelligence_service.get_categories") as mock_categories:
                    import pandas as pd
                    # Mock an empty DataFrame for startup_train so it runs in mock-free mode
                    mock_metrics.return_value = pd.DataFrame()
                    mock_summary.return_value = {"data_days": 100, "total_rows": 1000}
                    mock_categories.return_value = ["Pantolon", "Gömlek"]
                    
                    yield {
                        "metrics": mock_metrics,
                        "products": mock_products,
                        "summary": mock_summary,
                        "categories": mock_categories,
                    }

@pytest.fixture(autouse=True)
def mock_db_writer():
    """Mocks db.writer functions to prevent real DB inserts."""
    with patch("services.intelligence_service.save_predictions") as mock_save:
        with patch("services.intelligence_service.save_alert") as mock_alert:
            with patch("services.intelligence_service.db_get_alerts") as mock_get_alerts:
                mock_get_alerts.return_value = []
                yield {
                    "save_predictions": mock_save,
                    "save_alert": mock_alert,
                    "get_alerts": mock_get_alerts,
                }

@pytest.fixture(autouse=True)
def mock_scheduler():
    """Mocks APScheduler to prevent background tasks from running during tests."""
    with patch("app.create_scheduler") as mock_create:
        mock_instance = MagicMock()
        mock_create.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def client():
    """Returns a TestClient instance for FastAPI."""
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as c:
        yield c
