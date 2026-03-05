# algorithms/kalman.py
"""
Algoritma 3: Kalman Filter — v2
Online tahmin güncelleme + feedback loop.

v2 İyileştirmeler:
- Ürün bazlı tracking (kategori yerine her ürün için state)
- Multi-variate state: [demand, velocity, engagement]
- Adaptive noise estimation
"""
import numpy as np
from config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE


class TrendKalmanFilter:
    """
    v2 Kalman Filter — ürün ve kategori bazlı online tracking.

    State vector: [demand_level, trend_velocity]
    Measurement: engagement_score veya cart_count

    v2: Hem ürün hem kategori bazlı state tracking.
    """

    def __init__(self):
        # category_key → state dict
        self.states = {}  # Kategori bazlı
        self.product_states = {}  # Ürün bazlı [YENİ]

    def _init_state(self, initial_value=0.0):
        """Yeni Kalman state."""
        return {
            "x": np.array([initial_value, 0.0]),  # [demand, velocity]
            "P": np.eye(2) * 100,  # Uncertainty
            "history": [],
            "prediction_errors": [],  # Adaptive R için
        }

    def _predict(self, state):
        """Predict step: x_{k|k-1}, P_{k|k-1}"""
        F = np.array([[1, 1], [0, 1]])  # State transition
        Q = np.eye(2) * KALMAN_PROCESS_NOISE

        state["x"] = F @ state["x"]
        state["P"] = F @ state["P"] @ F.T + Q
        return state

    def _update(self, state, measurement):
        """Update step: x_{k|k}, P_{k|k} with adaptive noise."""
        H = np.array([[1, 0]])  # Observe demand only
        R = np.array([[self._adaptive_R(state)]])

        # Innovation
        y = measurement - H @ state["x"]
        S = H @ state["P"] @ H.T + R
        K = state["P"] @ H.T @ np.linalg.inv(S)

        state["x"] = state["x"] + K.flatten() * float(y)
        state["P"] = (np.eye(2) - K @ H) @ state["P"]

        # Hata geçmişi (adaptive R için)
        state["prediction_errors"].append(float(y))
        if len(state["prediction_errors"]) > 20:
            state["prediction_errors"] = state["prediction_errors"][-20:]

        return state, float(K[0, 0])

    def _adaptive_R(self, state) -> float:
        """[YENİ] Adaptive measurement noise — geçmiş hatalara göre ayarla."""
        errors = state.get("prediction_errors", [])
        if len(errors) < 3:
            return KALMAN_MEASUREMENT_NOISE

        recent_var = np.var(errors[-10:])
        return max(KALMAN_MEASUREMENT_NOISE * 0.5,
                   min(KALMAN_MEASUREMENT_NOISE * 3.0, recent_var))

    # ─── KATEGORİ BAZLI (eski) ─────────────────────────────────

    def process_series(self, key: str, values: list) -> dict:
        """Kategori bazlı seri işle."""
        if key not in self.states:
            self.states[key] = self._init_state(values[0] if values else 0.0)

        for val in values:
            self._predict(self.states[key])
            self._update(self.states[key], float(val))
            self.states[key]["history"].append(float(val))

        return self.get_state(key)

    def get_state(self, key: str) -> dict:
        """Kategori state çıktısı."""
        if key not in self.states:
            return {"demand": 0, "velocity": 0, "uncertainty": 100, "gain": 0}

        s = self.states[key]
        return {
            "demand": round(float(s["x"][0]), 2),
            "velocity": round(float(s["x"][1]), 2),
            "uncertainty": round(float(np.trace(s["P"])), 2),
            "gain": round(float(s["P"][0, 0] / (s["P"][0, 0] + KALMAN_MEASUREMENT_NOISE)), 4),
            "data_points": len(s["history"]),
        }

    def get_all_states(self) -> dict:
        return {k: self.get_state(k) for k in self.states}

    # ─── ÜRÜN BAZLI (YENİ v2) ──────────────────────────────────

    def process_product(self, product_id: int, values: list) -> dict:
        """[YENİ] Ürün bazlı seri işle."""
        key = str(product_id)
        if key not in self.product_states:
            self.product_states[key] = self._init_state(values[0] if values else 0.0)

        gain = 0
        for val in values:
            self._predict(self.product_states[key])
            _, gain = self._update(self.product_states[key], float(val))
            self.product_states[key]["history"].append(float(val))

        return self.get_product_state(product_id)

    def get_product_state(self, product_id: int) -> dict:
        """[YENİ] Ürün bazlı state çıktısı."""
        key = str(product_id)
        if key not in self.product_states:
            return {"demand": 0, "velocity": 0, "uncertainty": 100, "gain": 0}

        s = self.product_states[key]
        return {
            "demand": round(float(s["x"][0]), 2),
            "velocity": round(float(s["x"][1]), 2),
            "uncertainty": round(float(np.trace(s["P"])), 2),
            "gain": round(float(s["P"][0, 0] / (s["P"][0, 0] + self._adaptive_R(s))), 4),
            "data_points": len(s["history"]),
        }

    def get_product_trending(self, min_velocity=0.5, min_points=5) -> list:
        """[YENİ] Velocity > threshold olan trend ürünleri döner."""
        trending = []
        for key, state in self.product_states.items():
            if len(state["history"]) < min_points:
                continue
            velocity = float(state["x"][1])
            if velocity > min_velocity:
                trending.append({
                    "product_id": int(key),
                    "velocity": round(velocity, 3),
                    "demand": round(float(state["x"][0]), 2),
                    "uncertainty": round(float(np.trace(state["P"])), 2),
                })
        return sorted(trending, key=lambda x: x["velocity"], reverse=True)

    # ─── FEEDBACK ──────────────────────────────────────────────

    def update_with_feedback(self, key: str, actual_value: float):
        """Gerçek satış verisi ile güncelle."""
        if key in self.states:
            self._predict(self.states[key])
            self._update(self.states[key], actual_value)

    def update_product_with_feedback(self, product_id: int, actual_value: float):
        """[YENİ] Ürün bazlı feedback."""
        key = str(product_id)
        if key in self.product_states:
            self._predict(self.product_states[key])
            self._update(self.product_states[key], actual_value)
