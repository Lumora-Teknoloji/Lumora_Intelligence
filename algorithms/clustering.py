# algorithms/clustering.py
"""
Algoritma 6: K-Prototypes (Kümeleme)
Ürün segmentasyonu — sayısal + kategorik birlikte.
"""
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from config import N_CLUSTERS, MAX_ITER


class ProductClusterer:
    """
    Ürünleri performanslarına ve özelliklerine göre 4 kümeye ayırır.

    Küme 0 — Yıldızlar 🌟:    Yüksek engagement, yüksek trend
    Küme 1 — Nakit İnekler 💰: Stabil talep, düşük trend
    Küme 2 — Potansiyeller ❓:  Düşük satış, yükselen trend
    Küme 3 — Düşenler 🐕:      Düşük talep, düşen trend

    CatBoost'a feature: cluster_id
    İş kararı: "Yıldız kümesinin ortak özelliği: pamuk + siyah"
    """

    CLUSTER_LABELS = {0: "yildiz", 1: "nakit_inek", 2: "potansiyel", 3: "dusen"}

    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters or N_CLUSTERS
        self.model = None
        self.cluster_profiles = {}

    def fit(self, df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        """
        Ürünleri kümeler.

        Sayısal: ortalama engagement, cart_count, velocity, fiyat
        Kategorik: fabric, color, pattern, category

        Returns:
            DataFrame with added cluster_id column
        """
        # Her ürün için özet metrikler hesapla
        num_cols = ["engagement_score", "cart_count", "favorite_count", "price"]
        cat_cols = ["fabric", "color", "pattern", "category"]

        available_num = [c for c in num_cols if c in df.columns]
        available_cat = [c for c in cat_cols if c in df.columns]

        if not available_num:
            print("  ⚠ Clustering: Sayısal sütun bulunamadı")
            df["cluster_id"] = 0
            return df

        # Ürün bazlı özetler
        product_summary = df.groupby("product_id").agg(
            {col: "mean" for col in available_num}
        ).reset_index()

        # Kategorik bilgileri ekle (son kayıttaki)
        for col in available_cat:
            cat_values = df.groupby("product_id")[col].last().reset_index()
            product_summary = product_summary.merge(cat_values, on="product_id", how="left")

        # NaN temizlik
        for col in available_num:
            product_summary[col] = product_summary[col].fillna(0)
        for col in available_cat:
            if col in product_summary.columns:
                product_summary[col] = product_summary[col].fillna("unknown")

        if len(product_summary) < self.n_clusters:
            print(f"  ⚠ Clustering: Yeterli ürün yok ({len(product_summary)} < {self.n_clusters})")
            df["cluster_id"] = 0
            return df

        # Feature matrix
        feature_cols = available_num + [c for c in available_cat if c in product_summary.columns]
        X = product_summary[feature_cols].values

        # Kategorik sütun indeksleri
        cat_indices = list(range(len(available_num), len(feature_cols)))

        # Sayısal normalize et
        for i in range(len(available_num)):
            col_data = X[:, i].astype(float)
            std = col_data.std()
            if std > 0:
                X[:, i] = ((col_data - col_data.mean()) / std).astype(str)

        # K-Prototypes
        try:
            self.model = KPrototypes(
                n_clusters=self.n_clusters,
                max_iter=MAX_ITER,
                init="Cao",
                random_state=42,
            )
            clusters = self.model.fit_predict(X, categorical=cat_indices)

            product_summary["cluster_id"] = clusters

            # Küme profillerini çıkar
            self._build_profiles(product_summary, available_num, available_cat)

            # Ana DataFrame'e cluster_id ekle
            cluster_map = dict(zip(product_summary["product_id"], product_summary["cluster_id"]))
            df["cluster_id"] = df["product_id"].map(cluster_map).fillna(0).astype(int)

            if verbose:
                print(f"  ✓ Clustering: {len(product_summary)} ürün → {self.n_clusters} küme")
                for cid, profile in self.cluster_profiles.items():
                    label = self.CLUSTER_LABELS.get(cid, f"küme_{cid}")
                    count = profile.get("count", 0)
                    print(f"    Küme {cid} ({label}): {count} ürün")

        except Exception as e:
            print(f"  ⚠ Clustering hatası: {e}")
            df["cluster_id"] = 0

        return df

    def _build_profiles(self, summary_df, num_cols, cat_cols):
        """Her kümenin profilini çıkarır."""
        self.cluster_profiles = {}
        for cid in range(self.n_clusters):
            cluster_data = summary_df[summary_df["cluster_id"] == cid]
            if len(cluster_data) == 0:
                continue

            profile = {"count": len(cluster_data)}

            for col in num_cols:
                if col in cluster_data.columns:
                    profile[f"avg_{col}"] = round(float(cluster_data[col].mean()), 2)

            for col in cat_cols:
                if col in cluster_data.columns:
                    mode = cluster_data[col].mode()
                    profile[f"top_{col}"] = mode.iloc[0] if len(mode) > 0 else "unknown"

            self.cluster_profiles[cid] = profile

    def get_star_profile(self) -> dict:
        """Yıldız kümesinin profilini döner (envanter kararı için)."""
        if not self.cluster_profiles:
            return {}

        # En yüksek engagement'a sahip küme = yıldız
        best_cluster = max(
            self.cluster_profiles.items(),
            key=lambda x: x[1].get("avg_engagement_score", 0)
        )
        return {"cluster_id": best_cluster[0], **best_cluster[1]}
