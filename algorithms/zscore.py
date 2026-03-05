# algorithms/zscore.py
"""
Algoritma 1: Z-Score Anomali Tespiti
Hatalı veri filtreleme + viral ürün tespiti.
"""
import numpy as np
import pandas as pd
from config import ZSCORE_ERROR_THRESHOLD, ZSCORE_ANOMALY_THRESHOLD


class ZScoreDetector:
    """
    Her DailyMetric verisinin normallik skorunu hesaplar.

    Çıktılar:
        anomaly_flag: "normal" | "rising" | "dropping" | "error"
        z_cart: sepet sayısı Z-skoru
        z_engagement: engagement Z-skoru
        z_price: fiyat Z-skoru
    """

    def __init__(self):
        self.error_threshold = ZSCORE_ERROR_THRESHOLD       # |Z| > 3
        self.anomaly_threshold = ZSCORE_ANOMALY_THRESHOLD   # |Z| > 2.5

    def calculate_zscores(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        """
        DataFrame'deki belirli sütunlar için Z-Score hesaplar.
        Her ürün (product_id) kendi geçmişine göre değerlendirilir.
        """
        if columns is None:
            columns = ["cart_count", "engagement_score", "price"]

        result = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            z_col = f"z_{col}"
            result[z_col] = 0.0

            # Her ürün için ayrı Z-Score hesapla
            for pid in df["product_id"].unique():
                mask = df["product_id"] == pid
                series = df.loc[mask, col].astype(float)

                if len(series) < 3:
                    continue

                mean = series.mean()
                std = series.std()

                if std == 0:
                    result.loc[mask, z_col] = 0.0
                else:
                    result.loc[mask, z_col] = ((series - mean) / std).round(3)

        return result

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-Score hesapla + anomaly_flag ekle.

        Returns:
            DataFrame with added columns: z_cart_count, z_engagement_score, z_price, anomaly_flag
        """
        result = self.calculate_zscores(df, ["cart_count", "engagement_score", "price"])

        # Anomaly flag belirle
        flags = []
        for _, row in result.iterrows():
            z_cart = row.get("z_cart_count", 0)
            z_eng = row.get("z_engagement_score", 0)

            # En yüksek Z-Score'a bak
            max_z = max(abs(z_cart), abs(z_eng))

            if max_z > self.error_threshold:
                flags.append("error")
            elif z_cart > self.anomaly_threshold or z_eng > self.anomaly_threshold:
                flags.append("rising")     # Viral ürün adayı
            elif z_cart < -self.anomaly_threshold or z_eng < -self.anomaly_threshold:
                flags.append("dropping")   # Düşüş
            else:
                flags.append("normal")

        result["anomaly_flag"] = flags
        return result

    def get_viral_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-Score'u yüksek olan (viral başlangıcı) ürünleri döner."""
        analyzed = self.detect_anomalies(df)
        return analyzed[analyzed["anomaly_flag"] == "rising"]

    def filter_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hatalı verileri çıkarır, temiz veri döner."""
        analyzed = self.detect_anomalies(df)
        clean = analyzed[analyzed["anomaly_flag"] != "error"].copy()
        removed = len(df) - len(clean)
        if removed > 0:
            print(f"  ⚠ Z-Score: {removed} hatalı kayıt filtrelendi")
        return clean

    def summary(self, df: pd.DataFrame) -> dict:
        """Anomali özeti."""
        analyzed = self.detect_anomalies(df)
        counts = analyzed["anomaly_flag"].value_counts().to_dict()
        return {
            "total_records": len(analyzed),
            "normal": counts.get("normal", 0),
            "rising": counts.get("rising", 0),
            "dropping": counts.get("dropping", 0),
            "error": counts.get("error", 0),
        }
