# algorithms/prophet_model.py
"""
Algoritma 4: Prophet (Facebook/Meta)
Mevsimsel ayrıştırma — trend, mevsimsellik, tatil etkisi.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import PROPHET_CHANGEPOINT_SCALE, PROPHET_SEASONALITY_MODE


class SeasonalAnalyzer:
    """
    Prophet ile zaman serisi ayrıştırma.

    CatBoost'a 3 feature sağlar:
        trend_component: mevsimsellikten arındırılmış trend
        seasonal_component: saf mevsimsel etki
        seasonal_phase: mevsim döngüsünde konum (yükseliş/zirve/düşüş/dip)
    """

    def __init__(self):
        self.models = {}    # kategori → Prophet model

    def decompose(self, df: pd.DataFrame, category: str, metric_col="engagement_score") -> dict:
        """
        Bir kategori için mevsimsel ayrıştırma yapar.

        Args:
            df: DailyMetric (date, product_id, metric_col sütunları)
            category: Kategori adı
            metric_col: Ayrıştırılacak metrik

        Returns:
            trend, seasonal, residual bileşenleri + phase
        """
        # Kategori verisi
        cat_data = df[df["category"] == category].copy()
        if len(cat_data) < 14:
            return self._empty_result(len(df[df["category"] == category]))

        # Günlük ortalama (tüm ürünlerin ortalaması)
        daily = cat_data.groupby("date")[metric_col].mean().reset_index()
        daily.columns = ["ds", "y"]
        daily = daily.sort_values("ds")

        if len(daily) < 14:
            return self._empty_result(len(cat_data))

        try:
            from prophet import Prophet

            model = Prophet(
                changepoint_prior_scale=PROPHET_CHANGEPOINT_SCALE,
                seasonality_mode=PROPHET_SEASONALITY_MODE,
                daily_seasonality=False,
                yearly_seasonality=False,
                weekly_seasonality=True if len(daily) >= 14 else False,
            )
            model.fit(daily)
            self.models[category] = model

            # Bileşenleri çıkar
            forecast = model.predict(daily)

            trend = forecast["trend"].values
            weekly = forecast.get("weekly", pd.Series([0] * len(forecast))).values

            # Seasonal phase hesapla
            if len(weekly) > 1:
                diff = np.diff(weekly, prepend=weekly[0])
                phases = []
                for d, v in zip(diff, weekly):
                    if d > 0 and v > 0:
                        phases.append("rising")
                    elif d <= 0 and v > 0:
                        phases.append("peak")
                    elif d < 0 and v <= 0:
                        phases.append("falling")
                    else:
                        phases.append("trough")
            else:
                phases = ["stable"] * len(forecast)

            return {
                "trend": trend.tolist(),
                "seasonal": weekly.tolist(),
                "residual": (daily["y"].values - trend - weekly).tolist(),
                "phases": phases,
                "dates": daily["ds"].tolist(),
            }

        except Exception as e:
            print(f"  ⚠ Prophet hatası ({category}): {e}")
            return self._empty_result(len(cat_data))

    def add_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm kategoriler için Prophet ayrıştırma yapıp feature olarak ekler.

        Eklenen sütunlar: trend_component, seasonal_component, seasonal_phase
        """
        result = df.copy()
        result["trend_component"] = 0.0
        result["seasonal_component"] = 0.0
        result["seasonal_phase"] = "stable"

        for category in df["category"].unique():
            decomp = self.decompose(df, category)

            if not decomp["dates"]:
                continue

            # Date → trend/seasonal mapping
            date_trend = dict(zip(decomp["dates"], decomp["trend"]))
            date_season = dict(zip(decomp["dates"], decomp["seasonal"]))
            date_phase = dict(zip(decomp["dates"], decomp["phases"]))

            mask = result["category"] == category
            for idx in result[mask].index:
                d = result.loc[idx, "date"]
                if d in date_trend:
                    result.loc[idx, "trend_component"] = date_trend[d]
                    result.loc[idx, "seasonal_component"] = date_season[d]
                    result.loc[idx, "seasonal_phase"] = date_phase[d]

        return result

    def forecast(self, category: str, days=90) -> pd.DataFrame:
        """Geleceğe tahmin (Prophet native)."""
        if category not in self.models:
            return pd.DataFrame()

        model = self.models[category]
        future = model.make_future_dataframe(periods=days)
        return model.predict(future)

    def _empty_result(self, n):
        return {"trend": [], "seasonal": [], "residual": [], "phases": [], "dates": []}
