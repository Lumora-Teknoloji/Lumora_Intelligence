# algorithms/changepoint.py
"""
Algoritma 5: Change Point Detection (PELT)
Trend kırılma noktaları — mevsim geçişi, viral trend başlangıcı.
"""
import numpy as np
import pandas as pd
import ruptures as rpt
from config import CPD_PENALTY, CPD_MIN_SIZE


class ChangePointDetector:
    """
    Zaman serisindeki kalıcı rejim değişikliklerini tespit eder.

    Z-Score "bu değer normal mi?" → tekil anomali
    CPD "serinin karakteri ne zaman değişti?" → kalıcı geçiş

    CatBoost'a 2 feature sağlar:
        days_since_changepoint: son kırılmadan beri gün
        current_regime: yükseliş / düşüş / stabil
    """

    def __init__(self):
        self.penalty = CPD_PENALTY
        self.min_size = CPD_MIN_SIZE

    def detect(self, values: list, min_size=None) -> list:
        """
        Tek bir zaman serisinde kırılma noktalarını bulur.

        Args:
            values: Zaman sıralı değerler
            min_size: Minimum segment uzunluğu

        Returns:
            change_points: kırılma noktası indeksleri
        """
        if len(values) < (min_size or self.min_size) * 2:
            return []

        try:
            signal = np.array(values, dtype=float)
            # Sabit seri kontrolü
            if np.std(signal) < 1e-6:
                return []
            algo = rpt.Pelt(model="rbf", min_size=min_size or self.min_size)
            algo.fit(signal)
            change_points = algo.predict(pen=self._get_penalty(signal))

            # Son eleman (seri sonu) kaldır
            if change_points and change_points[-1] == len(values):
                change_points = change_points[:-1]

            return change_points
        except Exception:
            return []

    def detect_for_category(self, df: pd.DataFrame, category: str,
                             metric_col="engagement_score") -> dict:
        """
        Bir kategori için kırılma tespiti yapar.

        Returns:
            change_points: kırılma indeksleri
            change_dates: kırılma tarihleri
            current_regime: yükseliş / düşüş / stabil
            days_since_last: son kırılmadan beri gün
        """
        cat_data = df[df["category"] == category].copy()
        if len(cat_data) < self.min_size * 2:
            return {"change_points": [], "change_dates": [], "current_regime": "stable",
                    "days_since_last": len(cat_data)}

        daily = cat_data.groupby("date")[metric_col].mean().sort_index()
        values = daily.values.tolist()
        dates = daily.index.tolist()

        cps = self.detect(values)

        if not cps:
            return {
                "change_points": [],
                "change_dates": [],
                "current_regime": self._determine_regime(values, len(values)),
                "days_since_last": len(values),
            }

        last_cp = cps[-1]
        change_dates = [dates[cp] if cp < len(dates) else dates[-1] for cp in cps]

        # Son segment'in rejimi
        regime = self._determine_regime(values, last_cp)

        return {
            "change_points": cps,
            "change_dates": change_dates,
            "current_regime": regime,
            "days_since_last": len(values) - last_cp,
        }

    def add_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm kategoriler için CPD yapıp feature olarak ekler.

        Eklenen sütunlar: days_since_changepoint, current_regime
        """
        result = df.copy()
        result["days_since_changepoint"] = 0
        result["current_regime"] = "stable"

        for category in df["category"].unique():
            try:
                cpd_result = self.detect_for_category(df, category)
                mask = result["category"] == category
                result.loc[mask, "days_since_changepoint"] = cpd_result["days_since_last"]
                result.loc[mask, "current_regime"] = cpd_result["current_regime"]
            except Exception:
                pass  # Yeterli veri yok, varsayılan stable kalır

        return result

    def _determine_regime(self, values: list, split_point: int) -> str:
        """Son segment'in yönünü belirler."""
        if split_point >= len(values):
            segment = values
        else:
            segment = values[split_point:]

        if len(segment) < 3:
            return "stable"

        first_half = np.mean(segment[:len(segment)//2])
        second_half = np.mean(segment[len(segment)//2:])

        change_pct = (second_half - first_half) / (first_half + 1e-6) * 100

        if change_pct > 10:
            return "rising"
        elif change_pct < -10:
            return "falling"
        return "stable"

    def _get_penalty(self, signal):
        """BIC veya sabit penalty."""
        if self.penalty == "bic":
            return np.log(len(signal)) * np.var(signal)
        return float(self.penalty) if isinstance(self.penalty, (int, float)) else 10.0
