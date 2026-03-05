# engine/features.py
"""
Feature Mühendisliği Pipeline v2
Tüm veri kaynaklarından zengin feature matrix oluşturur.

v2 İyileştirmeler:
- 15+ JSONB attribute feature'ı (Sezon, Koleksiyon, Boy, Kalıp, Siluet...)
- rank_change_speed, size_depletion_rate, product_age
- day_of_week, is_weekend, discount_change
- stock_flip_count
"""
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Ham veriyi 40+ feature'lı zengin matrix'e dönüştürür.

    Feature grupları:
        - Rolling averages (7d, 14d, 30d)
        - Momentum (son 7 / önceki 7)
        - Velocity (günlük değişim)
        - Rank change speed (3 gün)
        - Size depletion rate
        - Product age
        - Day/weekend features
        - Discount change
        - Stock flip count
        - Price relative features
        - 15+ JSONB attribute features
    """

    # En önemli JSONB attribute'lar ve CatBoost'a feature olarak girecek olanlar
    JSONB_CATEGORICAL = [
        "Sezon", "Koleksiyon", "Boy", "Kalıp", "Siluet",
        "Yaka Tipi", "Kol Boyu", "Kol Tipi", "Bel", "Paça Tipi",
        "Desen", "Dokuma Tipi", "Persona", "Materyal Bileşeni",
        "Astar Durumu",
    ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm mühendislik feature'larını ekler."""
        result = df.copy()
        result = result.sort_values(["product_id", "date"])

        # Temel feature'lar
        result = self._add_rolling_averages(result)
        result = self._add_momentum(result)
        result = self._add_velocity(result)
        result = self._add_relative_features(result)

        # v2: Yeni feature'lar
        result = self._add_rank_change_speed(result)
        result = self._add_size_depletion(result)
        result = self._add_product_age(result)
        result = self._add_temporal_features(result)
        result = self._add_discount_change(result)
        result = self._add_stock_flip(result)
        result = self._add_jsonb_features(result)

        return result

    # ─── ROLLING AVERAGES ───────────────────────────────────────

    def _add_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """7, 14, 30 günlük hareketli ortalamalar."""
        for col in ["cart_count", "engagement_score", "favorite_count", "view_count"]:
            if col not in df.columns:
                continue
            for window in [7, 14, 30]:
                col_name = f"rolling_avg_{col}_{window}d"
                df[col_name] = df.groupby("product_id")[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

        # ── Rank rolling averages ────────────────────────────────
        # absolute_rank küçülüyor = ürün yükseliyor
        # rolling ortalamanın düşmesi = trend sinyali
        if "absolute_rank" in df.columns:
            for window in [3, 7, 14]:
                df[f"rolling_avg_rank_{window}d"] = df.groupby("product_id")["absolute_rank"].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            # Rank iyileşme hızı: negatif = rank düşüyor (iyi!)
            df["rank_velocity_7d"] = df.groupby("product_id")["absolute_rank"].transform(
                lambda x: x.diff(7).fillna(0)
            )
            df["rank_velocity_3d"] = df.groupby("product_id")["absolute_rank"].transform(
                lambda x: x.diff(3).fillna(0)
            )
            # Negatif = iyileşiyor, normalize et (-1000 ile +1000 klip)
            df["rank_momentum_score"] = (
                -df["rank_velocity_7d"].clip(-1000, 1000) / 1000.0
            ).round(4)

        # ── Growth Rate Feature'ları ─────────────────────────────
        # Mutlak değer değil, ORAN bazlı — geçmiş birikimden bağımsız
        if "favorite_count" in df.columns:
            rolling7  = df.groupby("product_id")["favorite_count"].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            rolling14_lag = df.groupby("product_id")["favorite_count"].transform(
                lambda x: x.shift(7).rolling(7, min_periods=1).mean()
            )
            # favorite_growth_14d: son 7 ort / önceki 7 ort  (standart pencere)
            df["favorite_growth_14d"] = (rolling7 / (rolling14_lag + 1e-6)).clip(0.05, 10.0).fillna(1.0).round(4)

            # favorite_growth_3d: son 3 gün / önceki 7 gün  (erken uyarı sinyali)
            # Küçük ama aniden yükselen ürünü 4 gün önceden yakalar
            rolling3 = df.groupby("product_id")["favorite_count"].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
            df["favorite_growth_3d"] = (rolling3 / (rolling14_lag + 1e-6)).clip(0.05, 15.0).fillna(1.0).round(4)

            # Spike dedektörü: 7 günde %200+ artış = olası viral
            prev7_fav = df.groupby("product_id")["favorite_count"].transform(
                lambda x: x.shift(7).fillna(x.iloc[0] if len(x) > 0 else 1)
            )
            df["fav_spike_7d"] = (
                (df["favorite_count"] - prev7_fav) / (prev7_fav + 1e-6)
            ).clip(-1.0, 20.0).fillna(0.0).round(3)
            df["is_fav_spike"] = (df["fav_spike_7d"] > 2.0).astype(int)  # %200+


        # ── [ADIM 1] Fiyat Değişim Feature'ı ───────────────────
        # Fiyat baskısı senaryosu: hızlı fiyat artışı → talep düşer
        if "price" in df.columns:
            df["price_change_7d"] = df.groupby("product_id")["price"].transform(
                lambda x: x.diff(7).fillna(0)
            )
            prev7_price = df.groupby("product_id")["price"].transform(
                lambda x: x.shift(7).fillna(x.iloc[0] if len(x) > 0 else 1)
            )
            df["price_change_pct"] = (
                df["price_change_7d"] / (prev7_price + 1e-6) * 100
            ).clip(-50, 100).fillna(0.0).round(2)
            df["price_rising_fast"] = (df["price_change_pct"] > 20).astype(int)

        return df

    # ─── MOMENTUM ───────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Son 7 gün / önceki 7 gün oranı."""
        for col in ["cart_count", "engagement_score", "favorite_count"]:
            if col not in df.columns:
                continue

            momentum_col = f"momentum_{col}_7d"
            df[momentum_col] = 1.0

            for pid in df["product_id"].unique():
                mask = df["product_id"] == pid
                series = df.loc[mask, col].values

                if len(series) < 14:
                    continue

                recent = np.mean(series[-7:])
                previous = np.mean(series[-14:-7])

                ratio = round(recent / (previous + 1e-6), 3)
                last_7_idx = df[mask].tail(7).index
                df.loc[last_7_idx, momentum_col] = ratio

        return df

    # ─── VELOCITY ───────────────────────────────────────────────

    def _add_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Günlük değişim hızı."""
        for col in ["cart_count", "engagement_score", "favorite_count"]:
            if col not in df.columns:
                continue
            vel_col = f"{col}_velocity"
            df[vel_col] = df.groupby("product_id")[col].transform(
                lambda x: x.diff().fillna(0)
            )
        return df

    # ─── RANK CHANGE SPEED ──────────────────────────────────────

    def _add_rank_change_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank değişim hızı ve momentumu — ana trend sinyali."""
        if "search_rank" in df.columns:
            df["rank_change_3d"] = df.groupby("product_id")["search_rank"].transform(
                lambda x: x.diff(periods=3).fillna(0)
            )
            df["rank_improving"] = (df["rank_change_3d"] < -3).astype(int)

        if "absolute_rank" in df.columns:
            # 3 / 7 günlük mutlak rank değişimi
            df["abs_rank_change_3d"] = df.groupby("product_id")["absolute_rank"].transform(
                lambda x: x.diff(periods=3).fillna(0)
            )
            df["abs_rank_change_7d"] = df.groupby("product_id")["absolute_rank"].transform(
                lambda x: x.diff(periods=7).fillna(0)
            )
            # Momentum: son 7 gün ortalaması / önceki 7 gün ortalaması
            # Değer < 1 = rank küçülüyor = ürün yükseliyor
            df["rank_momentum_ratio"] = df.groupby("product_id")["absolute_rank"].transform(
                lambda x: (
                    x.rolling(7, min_periods=1).mean() /
                    (x.shift(7).rolling(7, min_periods=1).mean() + 1e-6)
                ).fillna(1.0).clip(0.1, 10.0)
            )
            # < 0.8 = ciddi rank iyileşmesi (trend sinyali)
            df["rank_improving_strong"] = (df["rank_momentum_ratio"] < 0.8).astype(int)

        return df

    # ─── SIZE DEPLETION ─── [YENİ] ─────────────────────────────

    def _add_size_depletion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Beden tükenme hızı — güçlü talep sinyali."""
        # Tercih 1: total_stock integer kolonu (v3 sample_data)
        if "total_stock" in df.columns:
            stock_col = pd.to_numeric(df["total_stock"], errors="coerce").fillna(0)
            first_stock = df.groupby("product_id")["total_stock"].transform(
                lambda x: pd.to_numeric(x, errors="coerce").fillna(0).iloc[0]
            )
            df["size_depletion_rate"] = (
                (first_stock - stock_col) / (first_stock + 1e-6)
            ).clip(0, 1)
            df["rapid_size_depletion"] = (df["size_depletion_rate"] > 0.3).astype(int)
            df["size_change_velocity"] = df.groupby("product_id")["total_stock"].transform(
                lambda x: pd.to_numeric(x, errors="coerce").fillna(0).diff().fillna(0)
            )
            return df

        # Tercih 2: available_sizes JSON string (eski format — sayısal ise)
        if "available_sizes" not in df.columns:
            return df

        def safe_total(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                import json
                try:
                    d = json.loads(val)
                    if isinstance(d, dict):
                        return float(sum(d.values()))
                    return float(val)
                except Exception:
                    return 0.0
            return 0.0

        stock_series = df["available_sizes"].apply(safe_total)
        first_stock = df.groupby("product_id")["available_sizes"].transform(
            lambda x: x.apply(safe_total).iloc[0]
        )
        df["size_depletion_rate"] = (
            (first_stock - stock_series) / (first_stock + 1e-6)
        ).clip(0, 1)
        df["rapid_size_depletion"] = (df["size_depletion_rate"] > 0.3).astype(int)
        df["size_change_velocity"] = df.groupby("product_id")["available_sizes"].transform(
            lambda x: x.apply(safe_total).diff().fillna(0)
        )

        return df

    # ─── PRODUCT AGE ─── [YENİ] ────────────────────────────────

    def _add_product_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ürünün piyasadaki yaşı (gün)."""
        if "first_seen_at" in df.columns:
            df["product_age_days"] = (
                pd.to_datetime(df["date"]) - pd.to_datetime(df["first_seen_at"])
            ).dt.days.fillna(0).astype(int)
        else:
            # first_seen_at yoksa, veri içindeki ilk tarih
            first_dates = df.groupby("product_id")["date"].transform("min")
            df["product_age_days"] = (
                pd.to_datetime(df["date"]) - pd.to_datetime(first_dates)
            ).dt.days.fillna(0).astype(int)

        # Yeni ürün flag'i (7 günden genç)
        df["is_new_product"] = (df["product_age_days"] <= 7).astype(int)

        return df

    # ─── TEMPORAL FEATURES ─── [YENİ] ──────────────────────────

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gün ve hafta sonu feature'ları."""
        dates = pd.to_datetime(df["date"])
        df["day_of_week"] = dates.dt.dayofweek  # 0=Pazartesi, 6=Pazar
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        df["month"] = dates.dt.month

        return df

    # ─── DISCOUNT CHANGE ─── [YENİ] ───────────────────────────

    def _add_discount_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """İndirim değişim hızı."""
        if "discount_rate" not in df.columns:
            return df

        df["discount_change"] = df.groupby("product_id")["discount_rate"].transform(
            lambda x: x.diff().fillna(0)
        )
        # İndirim artışı (negatif fiyat sinyali ama talep artışı)
        df["discount_increasing"] = (df["discount_change"] > 5).astype(int)

        # İndirim yoğunluğu kategorik
        df["discount_intensity"] = np.where(
            df["discount_rate"] > 30, "high",
            np.where(df["discount_rate"] > 15, "medium", "low")
        )

        return df

    # ─── STOCK FLIP ─── [YENİ] ─────────────────────────────────

    def _add_stock_flip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stok durumu değişim sayısı (stokta↔stok dışı)."""
        if "stock_status" not in df.columns:
            return df

        df["stock_flip_count"] = df.groupby("product_id")["stock_status"].transform(
            lambda x: (x != x.shift()).cumsum()
        )

        # Şu an stok dışı mı
        df["is_out_of_stock"] = (~df["stock_status"].astype(bool)).astype(int)

        return df

    # ─── RELATIVE FEATURES ─────────────────────────────────────

    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategori ortalamasına göre göreceli metrikler."""
        if "price" in df.columns and "category" in df.columns:
            cat_avg_price = df.groupby("category")["price"].transform("mean")
            df["price_vs_category_avg"] = np.where(
                cat_avg_price > 0,
                (df["price"] / cat_avg_price).round(3),
                1.0
            )

        if "engagement_score" in df.columns and "category" in df.columns:
            cat_avg_eng = df.groupby("category")["engagement_score"].transform("mean")
            df["engagement_vs_category"] = np.where(
                cat_avg_eng > 0,
                (df["engagement_score"] / cat_avg_eng).round(3),
                1.0
            )

        if "cart_count" in df.columns and "category" in df.columns:
            cat_avg_cart = df.groupby("category")["cart_count"].transform("mean")
            df["cart_vs_category"] = np.where(
                cat_avg_cart > 0,
                (df["cart_count"] / cat_avg_cart).round(3),
                1.0
            )

        return df

    # ─── JSONB ATTRIBUTE FEATURES ─── [YENİ] ───────────────────

    def _add_jsonb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """JSONB attribute'larını feature olarak ekler."""
        for attr in self.JSONB_CATEGORICAL:
            col_name = f"attr_{attr.lower().replace(' ', '_').replace('/', '_')}"
            if attr in df.columns:
                df[col_name] = df[attr].fillna("unknown").astype(str)
            elif col_name in df.columns:
                df[col_name] = df[col_name].fillna("unknown").astype(str)
            # Eğer sütun yoksa, _add_jsonb_features öncesinde
            # test_real_data.py'de SQL sorgusu ile çekilmiş olmalı

        return df

    # ─── SUMMARY ────────────────────────────────────────────────

    def get_feature_summary(self, df: pd.DataFrame) -> dict:
        """Feature matrix özeti."""
        numeric_features = [c for c in df.columns if
                           c.startswith("rolling_") or c.startswith("momentum_") or
                           c.endswith("_velocity") or c.startswith("z_") or
                           c in ["cluster_id", "days_since_changepoint",
                           "trend_component", "seasonal_component",
                           "visual_trend_score", "price_vs_category_avg",
                           "engagement_vs_category", "cart_vs_category",
                           "rank_change_3d", "abs_rank_change_3d",
                           "rank_improving", "size_depletion_rate",
                           "rapid_size_depletion", "size_change_velocity",
                           "product_age_days", "is_new_product",
                           "day_of_week", "is_weekend", "week_of_year", "month",
                           "discount_change", "discount_increasing",
                           "stock_flip_count", "is_out_of_stock"]]

        categorical_features = [c for c in df.columns if
                               c.startswith("attr_") or
                               c in ["anomaly_flag", "current_regime",
                               "discount_intensity", "seasonal_phase"]]

        return {
            "numeric_features": len(numeric_features),
            "categorical_features": len(categorical_features),
            "total_features": len(numeric_features) + len(categorical_features),
            "numeric_names": sorted(numeric_features),
            "categorical_names": sorted(categorical_features),
            "rows": len(df),
            "products": df["product_id"].nunique(),
        }
