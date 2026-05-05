import pytest
from unittest.mock import patch
import os

def test_health_check(client):
    """Test the /health endpoint (public)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "engine_trained" in data

def test_auth_missing_key(client):
    """Test that endpoints other than /health require X-Internal-Key."""
    # Ensure INTERNAL_API_KEY is strict
    os.environ["INTERNAL_API_KEY"] = "test-internal-key"
    
    response = client.get("/predict")
    assert response.status_code == 401
    assert response.json()["detail"] == "Geçersiz iç API anahtarı"

def test_auth_invalid_key(client):
    """Test that invalid X-Internal-Key is rejected."""
    response = client.get(
        "/predict",
        headers={"X-Internal-Key": "wrong-key"}
    )
    assert response.status_code == 401

def test_predict_endpoint_success(client):
    """Test /predict endpoint with valid auth and mocked engine."""
    mock_predictions = [
        {
            "product_id": 1,
            "trend_label": "TREND",
            "trend_score": 95.5,
            "confidence": 0.9,
            "name": "Test Pantolon",
            "price": 200.0,
        }
    ]
    
    with patch("services.intelligence_service.intelligence_service.predict") as mock_predict:
        mock_predict.return_value = mock_predictions
        
        response = client.get(
            "/predict?category=Pantolon&top_n=10",
            headers={"X-Internal-Key": "test-internal-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["trend_label"] == "TREND"
        assert data["count"] == 1
        
        mock_predict.assert_called_once_with(category="Pantolon", top_n=10)

def test_analyze_endpoint_success(client):
    """Test /analyze endpoint for a specific product."""
    mock_analysis = {
        "product_id": 123,
        "trend_label": "POTANSIYEL",
        "trend_score": 80.0,
        "confidence": 0.85,
        "signals": {},
        "data_points": 90
    }
    
    with patch("services.intelligence_service.intelligence_service.analyze") as mock_analyze:
        mock_analyze.return_value = mock_analysis
        
        response = client.post(
            "/analyze",
            json={"product_id": 123},
            headers={"X-Internal-Key": "test-internal-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == 123
        assert data["trend_label"] == "POTANSIYEL"
        
        mock_analyze.assert_called_once_with(123)
