# engine/predictor.py
"""
Ensemble Predictor v2 — adaptive ağırlıklı, 8 algoritma.

v2 İyileştirmeler:
- Adaptive ensemble ağırlıkları (Bayesian Optimization)
- Ürün bazlı Kalman + kategori bazlı Kalman
- SHAP raporu
- Accuracy metrikleri
- Zengin status raporu
"""
import pandas as pd
import numpy as np
from config import ENSEMBLE_CATBOOST_WEIGHT, ENSEMBLE_KALMAN_WEIGHT

from algorithms.zscore import ZScoreDetector
from algorithms.kalman import TrendKalmanFilter
from algorithms.catboost_model import DemandPredictor
from algorithms.changepoint import ChangePointDetector
from algorithms.clustering import ProductClusterer
from engine.features import FeatureEngineer


class PredictionEngine:
    """
    v2 Ensemble: CatBoost × w1 + Kalman(ürün) × w2 + Kalman(kategori) × w3
    Adaptive ağırlıklar feedback loop ile güncellenir.
    """

    def __init__(self, use_prophet=False, use_clip=False):
        self.zscore = ZScoreDetector()
        self.feature_eng = FeatureEngineer()
        self.clusterer = ProductClusterer()
        self.cpd = ChangePointDetector()
        self.catboost = DemandPredictor()
        self.kalman = TrendKalmanFilter()

        # v2: Adaptive ağırlıklar
        self.weights = {
            "catboost": ENSEMBLE_CATBOOST_WEIGHT,
            "kalman_product": ENSEMBLE_KALMAN_WEIGHT * 0.6,
            "kalman_category": ENSEMBLE_KALMAN_WEIGHT * 0.4,
        }
        self.feedback_history = []

        # ── Feedback ceza sistemi ──────────────────────────────────
        # {product_id: penalty_multiplier}  → 0.0=tamamen cezalı, 1.0=cezasız
        self._feedback_penalties: dict = {}

        # ── Tahmin Geçmişi (Batch Feedback için) ──────────────────
        # {period_key: {"TREND": [pid,...], "POTANSIYEL": [...], predictions_df: ...}}
        # Son 12 hafta tutulur, eskisi silinir
        self._prediction_history: dict = {}
        self._predict_call_count: int = 0

        self.prophet = None
        self.clip = None
        self.use_prophet = use_prophet
        self.use_clip = use_clip

        if use_prophet:
            from algorithms.prophet_model import SeasonalAnalyzer
            self.prophet = SeasonalAnalyzer()

        if use_clip:
            from algorithms.clip_model import VisualMatcher
            self.clip = VisualMatcher()

        self.is_trained = False

    def train(self, df: pd.DataFrame, verbose=True):
        """v2: Tam pipeline eğitimi."""
        if verbose:
            print("=" * 60)
            print("TREND TAHMIN MOTORU v2 -- EGITIM")
            print("=" * 60)
            print(f"  Veri: {len(df)} kayit, {df['product_id'].nunique()} urun")
            print()

        # date kolonu: string veya datetime olabilir — normalize et
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Katman 2: Z-Score
        if verbose:
            print("[Katman 2] Z-Score veri temizleme")
        df_clean = self.zscore.filter_errors(df)
        df_clean = self.zscore.detect_anomalies(df_clean)
        summary = self.zscore.summary(df)
        if verbose:
            print(f"  Normal: {summary['normal']}, Rising: {summary['rising']}, "
                  f"Error: {summary['error']}")
            print()

        # Katman 3a: Feature muhendisligi
        if verbose:
            print("[Katman 3a] Feature muhendisligi v2")
        df_feat = self.feature_eng.build_features(df_clean)
        feat_summary = self.feature_eng.get_feature_summary(df_feat)
        if verbose:
            print(f"  {feat_summary['total_features']} feature olusturuldu "
                  f"({feat_summary['numeric_features']} sayisal + "
                  f"{feat_summary['categorical_features']} kategorik)")
            print()

        # Katman 3b-c: CPD ve K-Prototypes — devre dışı bırakıldı
        # - CPD: change point feature'ları CatBoost'a zaten giriş olarak verilebilir
        # - K-Prototypes: kategori zaten CatBoost'ta kategorik feature
        # - İkisi de somut doğruluk kazanımı sağlamadı, complexity ekliyordu

        # Katman 3d: Prophet (opsiyonel)
        if self.use_prophet and self.prophet:
            if verbose:
                print("[Katman 3d] Prophet mevsimsel ayristirma")
            df_feat = self.prophet.add_features_to_df(df_feat)
            if verbose:
                print("  Mevsimsel feature'lar eklendi")
                print()

        # Katman 4a: CatBoost
        if verbose:
            print("[Katman 4a] CatBoost v2 egitimi")
        success = self.catboost.train(df_feat, verbose=verbose)
        if not success:
            print("  CatBoost egitilemedi!")
            return
        if verbose:
            print()

        # Katman 4b: Kalman — kategori + urun bazli
        if verbose:
            print("[Katman 4b] Kalman Filter v2 (urun + kategori)")

        # Kategori bazli
        for category in df_feat["category"].unique():
            cat_data = df_feat[df_feat["category"] == category]
            daily_eng = cat_data.groupby("date")["engagement_score"].mean().sort_index().values
            self.kalman.process_series(category, daily_eng.tolist())

        # Urun bazli [YENi]
        product_count = 0
        for pid in df_feat["product_id"].unique():
            p_data = df_feat[df_feat["product_id"] == pid].sort_values("date")
            if len(p_data) < 3:
                continue
            cart_series = p_data["cart_count"].values.tolist()
            self.kalman.process_product(pid, cart_series)
            product_count += 1

        if verbose:
            trending = self.kalman.get_product_trending(min_velocity=0.3)
            print(f"  {product_count} urun icin Kalman baslatildi")
            print(f"  Trend tespit: {len(trending)} urun (velocity > 0.3)")
            for t in trending[:5]:
                print(f"    #{t['product_id']}: velocity={t['velocity']:.2f}, "
                      f"demand={t['demand']:.1f}")
            print()

        self.is_trained = True
        self._last_training_data = df_feat

        if verbose:
            print("=" * 60)
            print("EGITIM TAMAMLANDI")
            print("=" * 60)

    # Varsayilan tuning parametreleri
    DEFAULT_TUNING = {
        "score_weights": {"growth": 0.40, "velocity": 0.35, "demand": 0.25},
        "clip_ranges": {"growth": (-100, 500), "velocity": (-5, 10)},
        "label_thresholds": {"trend": 70, "potansiyel": 40, "stabil": 20},
    }

    def predict(self, df: pd.DataFrame = None, tuning_params: dict = None,
                category_tuning: dict = None) -> pd.DataFrame:
        """v3: CatBoost + Kalman + Growth Rate = Trend Score.

        tuning_params: global override (tüm kategoriler için)
        category_tuning: kategori bazlı override dict
            {"crop": {"score_weights": {...}, ...}, "tayt": {...}, ...}

        Öncelik: category_tuning > tuning_params > DEFAULT_TUNING
        """
        if not self.is_trained:
            print("Motor henuz egitilmedi!")
            return pd.DataFrame()

        # Tuning parametre çözücü: category_tuning > tuning_params > DEFAULT_TUNING
        def resolve_params(category=None):
            tp = {}
            for key, default in self.DEFAULT_TUNING.items():
                # 1. Category-specific override
                if category_tuning and category in category_tuning and key in category_tuning[category]:
                    tp[key] = category_tuning[category][key]
                # 2. Global override
                elif tuning_params and key in tuning_params:
                    tp[key] = tuning_params[key]
                # 3. Default
                else:
                    tp[key] = default
            return tp

        data = df if df is not None else self._last_training_data

        # CatBoost tahminleri
        cb_predictions = self.catboost.predict(data)
        if cb_predictions.empty:
            return pd.DataFrame()

        # Kalman tahminleri
        kalman_cat_results = []
        for category in data["category"].unique():
            state = self.kalman.get_state(category)
            kalman_cat_results.append({
                "category": category,
                "kalman_cat_demand": state["demand"],
                "kalman_cat_velocity": state["velocity"],
            })
        kalman_cat_df = pd.DataFrame(kalman_cat_results)

        kalman_prod_results = []
        for pid in data["product_id"].unique():
            state = self.kalman.get_product_state(pid)
            kalman_prod_results.append({
                "product_id": pid,
                "kalman_prod_demand": state["demand"],
                "kalman_prod_velocity": state["velocity"],
            })
        kalman_prod_df = pd.DataFrame(kalman_prod_results)

        # Cart buyume oranini hesapla (ilk vs son dönem)
        growth_results = []
        for pid in data["product_id"].unique():
            pdata = data[data["product_id"] == pid].sort_values("date")
            n = len(pdata)
            if n < 6:
                growth_results.append({"product_id": pid, "cart_growth_pct": 0.0})
                continue
            first = pdata.head(max(3, n // 3))["cart_count"].mean()
            last = pdata.tail(max(3, n // 3))["cart_count"].mean()
            growth = (last - first) / (first + 1e-6) * 100
            growth_results.append({"product_id": pid, "cart_growth_pct": round(growth, 1)})
        growth_df = pd.DataFrame(growth_results)

        # Birlestir
        product_cats = data.groupby("product_id")["category"].first().reset_index()
        result = cb_predictions.merge(product_cats, on="product_id", how="left")
        result = result.merge(kalman_cat_df, on="category", how="left")
        result = result.merge(kalman_prod_df, on="product_id", how="left")
        result = result.merge(growth_df, on="product_id", how="left")

        # ── TREND SCORE — KATEGORİ BAZLI ─────────────────────────
        def norm(s):
            r = s.max() - s.min()
            if r < 1e-6:
                return pd.Series(0.5, index=s.index)
            return (s - s.min()) / r

        # Her kategori kendi parametreleri ile skorlanır
        scored_parts = []
        for cat in result["category"].unique():
            mask = result["category"] == cat
            cat_result = result[mask].copy()
            tp = resolve_params(cat)
            sw = tp["score_weights"]
            cr = tp["clip_ranges"]

            vel = cat_result["kalman_prod_velocity"].fillna(0)
            gr  = cat_result["cart_growth_pct"].fillna(0)
            dem = cat_result["predicted_demand"].fillna(0)

            score_g = norm(gr.clip(lower=cr["growth"][0], upper=cr["growth"][1]))
            score_v = norm(vel.clip(lower=cr["velocity"][0], upper=cr["velocity"][1]))
            score_d = norm(dem.clip(lower=0))

            # [FIX] CatBoost composite score'u direkt ekle — bu zaten yükseliş/düşüş
            # öğreti. Normalizasyon bunu gizliyordu, direkt normalized olarak dahil et.
            if "predicted_demand" in cat_result.columns:
                # CatBoost'un ham tahmini → normalize et
                raw_cb = cat_result["predicted_demand"].fillna(0)
                score_cb = norm(raw_cb.clip(lower=0))
            else:
                score_cb = pd.Series(0.5, index=cat_result.index)

            # favorite_growth_14d varsa onu da ekle (değişim bazlı sinyal)
            # favorite_growth: 14d standart + 3d erken sinyal — kombinasyonu kullan
            if "favorite_growth_14d" in cat_result.columns:
                fg14 = cat_result["favorite_growth_14d"].fillna(1.0)
                score_fav14 = norm(fg14.clip(0.05, 10.0))
            else:
                score_fav14 = pd.Series(0.5, index=cat_result.index)

            if "favorite_growth_3d" in cat_result.columns:
                fg3 = cat_result["favorite_growth_3d"].fillna(1.0)
                score_fav3 = norm(fg3.clip(0.05, 15.0))
            else:
                score_fav3 = score_fav14

            # Kombine: erken sinyal %60, standart %40 (erken uyarıya ağırlık ver)
            score_fav = score_fav3 * 0.60 + score_fav14 * 0.40

            # Ağırlıklar — relative growth öncelikli:
            # fav_growth  %45 → değişim oranı (küçük ama hızlı ürünü yakalar)
            # velocity    %25 → Kalman erken sinyal
            # CatBoost    %20 → mutlak talep (azaltıldı, Pareto'yu bastırır)
            # cart_growth %10 → bonus sinyal
            cat_result["trend_score"] = (
                score_fav * 0.45 +
                score_v   * 0.25 +
                score_cb  * 0.20 +
                score_g   * 0.10
            ) * 100
            cat_result["trend_score"] = cat_result["trend_score"].round(1)



            # Confidence
            cb_rank = dem.rank(ascending=False, method="min")
            kl_rank = vel.rank(ascending=False, method="min")
            gr_rank = gr.rank(ascending=False, method="min")
            nc = len(cat_result)
            if nc > 1:
                rank_diff = (abs(cb_rank - kl_rank) + abs(cb_rank - gr_rank) + abs(kl_rank - gr_rank)) / 3
                agreement = 1 - (rank_diff / nc)
                cat_result["confidence"] = (agreement.clip(0, 1) * 100).round(0).astype(int)
            else:
                cat_result["confidence"] = 50

            # ── [ADIM 3] Hybrid Quantile Eşikler ─────────────────
            # Quantile + minimum floor — baseline gerileme sorununu çözer
            scores = cat_result["trend_score"]
            if nc >= 6:
                p85 = max(scores.quantile(0.85), 62.0)  # floor: en az 62
                p55 = max(scores.quantile(0.55), 42.0)  # floor: en az 42
                p25 = max(scores.quantile(0.25), 22.0)  # floor: en az 22
            elif nc >= 4:
                p85 = max(scores.quantile(0.80), 60.0)
                p55 = max(scores.quantile(0.50), 40.0)
                p25 = max(scores.quantile(0.25), 20.0)
            else:
                p85, p55, p25 = 70.0, 45.0, 25.0

            cat_result["trend_label"] = np.where(
                scores >= p85, "TREND",
                np.where(scores >= p55, "POTANSIYEL",
                np.where(scores >= p25, "STABIL", "DUSEN")))

            # ── [ADIM 4] Aktif DUSEN Override ─────────────────────
            # Rank hızla kötüleşiyor VE favori azalıyorsa → DUSEN
            if "abs_rank_change_7d" in cat_result.columns and \
               "favorite_growth_14d" in cat_result.columns:
                rank_worsening = cat_result["abs_rank_change_7d"] > 300  # 300+ pozisyon kaybı
                fav_declining  = cat_result["favorite_growth_14d"] < 0.80  # %20+ favori kaybı
                dusen_override = rank_worsening & fav_declining
                cat_result.loc[dusen_override, "trend_label"] = "DUSEN"

            # ── [ADIM 5] Viral Spike Override ─────────────────────
            # 7 günde %200+ favori artışı VE rank iyileşiyorsa → TREND
            if "is_fav_spike" in cat_result.columns and \
               "rank_improving_strong" in cat_result.columns:
                spike_mask = (
                    (cat_result["is_fav_spike"] == 1) &
                    (cat_result["rank_improving_strong"] == 1)
                )
                cat_result.loc[spike_mask, "trend_label"] = "TREND"

            scored_parts.append(cat_result)



        result = pd.concat(scored_parts, ignore_index=True)

        # Ensemble demand
        w = self.weights
        cb_d = result["predicted_demand"]
        kp_d = result["kalman_prod_demand"].fillna(cb_d)
        kc_d = result["kalman_cat_demand"].fillna(cb_d)
        result["ensemble_demand"] = np.round(
            cb_d * w["catboost"] + kp_d * w["kalman_product"] + kc_d * w["kalman_category"]
        ).clip(lower=0).astype(int)

        # ── Feedback cezalarını uygula ─────────────────────────────
        # Normalizasyon sonrası direkt çarpar → kategoriden bağımsız
        if self._feedback_penalties:
            result["feedback_penalty"] = result["product_id"].map(
                lambda pid: self._feedback_penalties.get(int(pid), 1.0)
            )
            result["trend_score"] = (
                result["trend_score"] * result["feedback_penalty"]
            ).round(1)
            result["ensemble_demand"] = (
                result["ensemble_demand"] * result["feedback_penalty"]
            ).clip(lower=0).round(0).astype(int)

            # Etiketi de güncelle (skor değişti)
            lt_default = self.DEFAULT_TUNING["label_thresholds"]
            result["trend_label"] = np.where(
                result["trend_score"] >= lt_default["trend"],     "TREND",
                np.where(result["trend_score"] >= lt_default["potansiyel"], "POTANSIYEL",
                np.where(result["trend_score"] >= lt_default["stabil"],     "STABIL",
                "DUSEN")))

        columns = ["product_id", "category",
                   "trend_score", "trend_label", "confidence",
                   "cart_growth_pct", "kalman_prod_velocity",
                   "predicted_demand", "ensemble_demand",
                   "kalman_prod_demand", "kalman_cat_demand"]
        final = result[[c for c in columns if c in result.columns]].sort_values(
            "trend_score", ascending=False)

        # ── Tahmin snapshot'u kaydet (batch feedback için) ───────────
        self._predict_call_count += 1
        period_key = f"period_{self._predict_call_count}"
        label_groups: dict = {}
        for lbl in ["TREND", "POTANSIYEL", "STABIL", "DUSEN"]:
            pids = final[final["trend_label"] == lbl]["product_id"].tolist()
            demands = final[final["trend_label"] == lbl]["ensemble_demand"].tolist()
            label_groups[lbl] = {
                "product_ids": [int(p) for p in pids],
                "predicted_demands": [int(d) for d in demands],
                "total_predicted": int(sum(demands)),
            }
        self._prediction_history[period_key] = label_groups
        # Son 12 tahmin tut, eskisini sil
        if len(self._prediction_history) > 12:
            oldest = min(self._prediction_history.keys())
            del self._prediction_history[oldest]

        return final

    def feedback(self, category: str, actual_sales: int, predicted_demand: int = None,
                 product_id: int = None):
        """v2: Feedback + ceza sistemi + adaptive ağırlık güncelleme."""
        # Kalman guncelle
        self.kalman.update_with_feedback(category, float(actual_sales))
        if product_id:
            self.kalman.update_product_with_feedback(product_id, float(actual_sales))

        # ── Ceza hesapla ──────────────────────────────────────────
        # Hata oranı: predicted 150, actual 12 → hata %92
        # Eşik %70 üzerindeyse direkt trend_score'a ceza uygula
        if predicted_demand and product_id:
            accuracy = min(actual_sales, predicted_demand) / (max(actual_sales, predicted_demand) + 1e-6)
            error_rate = 1.0 - accuracy  # 0=mükemmel, 1=tamamen yanlış

            if error_rate > 0.70:
                # %70+ hata → ciddi ceza (trend_score * 0.35)
                penalty = 0.35
            elif error_rate > 0.50:
                # %50-70 hata → orta ceza
                penalty = 0.55
            elif error_rate > 0.30:
                # %30-50 hata → hafif ceza
                penalty = 0.75
            else:
                # %30 altı hata → ceza yok (normal sapma)
                penalty = 1.0

            if penalty < 1.0:
                pid_key = int(product_id)
                # Mevcut cezayla birleştir (tekrar yanılırsa daha fazla ceza)
                existing = self._feedback_penalties.get(pid_key, 1.0)
                self._feedback_penalties[pid_key] = round(existing * penalty, 3)

        # v2: Feedback geçmişine kaydet
        if predicted_demand:
            error = abs(actual_sales - predicted_demand)
            accuracy_val = round(min(actual_sales, predicted_demand) /
                                  (max(actual_sales, predicted_demand) + 1e-6), 3)
            self.feedback_history.append({
                "category": category,
                "product_id": product_id,
                "predicted": predicted_demand,
                "actual": actual_sales,
                "error": error,
                "accuracy": accuracy_val,
            })

        # Adaptive ağırlık güncelleme (her 10 feedback'te)
        if len(self.feedback_history) % 10 == 0 and len(self.feedback_history) >= 10:
            self._update_weights()

        state = self.kalman.get_state(category)
        return {
            "category": category,
            "actual_sales": actual_sales,
            "predicted_demand": predicted_demand,
            "updated_kalman_state": state,
            "penalty_applied": self._feedback_penalties.get(int(product_id), 1.0) if product_id else 1.0,
        }

    # ──────────────────────────────────────────────────────────────
    # BATCH FEEDBACK — Kullanıcı sadece toplam satışı söyler
    # ──────────────────────────────────────────────────────────────

    def feedback_by_label(self, label: str, total_sold: int,
                          period_key: str = None) -> dict:
        """
        "TREND dediğin ürünler toplam X sattı" → sistem öğrenir.

        Args:
            label:       "TREND", "POTANSIYEL", "STABIL", "DUSEN"
            total_sold:  Gerçekte o etiketli ürünlerin toplam satışı
            period_key:  Hangi tahmin dönemini kullan (None=son tahmin)

        Returns:
            dict ile özet (penalty, hata oranı vb.)
        """
        # En son (veya belirtilen) tahmin dönemini bul
        if not self._prediction_history:
            return {"error": "Henüz tahmin yapılmadı"}

        if period_key is None:
            period_key = max(self._prediction_history.keys())

        snapshot = self._prediction_history.get(period_key, {})
        group = snapshot.get(label, {})

        if not group or not group.get("product_ids"):
            return {"error": f"'{label}' grubunda ürün bulunamadı ({period_key})"}

        product_ids     = group["product_ids"]
        pred_demands    = group["predicted_demands"]
        total_predicted = group["total_predicted"]
        n_products      = len(product_ids)

        # Hata oranı (grup bazlı)
        if total_predicted > 0:
            error_rate = abs(total_sold - total_predicted) / total_predicted
        else:
            error_rate = 1.0

        # Grup bazlı ceza / ödül
        if total_sold < total_predicted:
            # Az sattı → ceza
            if error_rate > 0.60:
                group_penalty = 0.40
            elif error_rate > 0.40:
                group_penalty = 0.60
            elif error_rate > 0.20:
                group_penalty = 0.80
            else:
                group_penalty = 1.0    # %20 altı fark → ceza yok
        else:
            # Beklenenden fazla sattı → ödül (ceza kaldır)
            group_penalty = min(1.0, 1.0 + (total_sold - total_predicted) / (total_predicted + 1e-6) * 0.3)

        # Her ürüne orantılı gerçek satış tahmin et (proportional allocation)
        applied_penalties = {}
        for pid, pred in zip(product_ids, pred_demands):
            if total_predicted > 0:
                estimated_actual = int(total_sold * (pred / total_predicted))
            else:
                estimated_actual = total_sold // n_products

            # Kalman'ı güncelle (kategori bazlı)
            self.kalman.update_product_with_feedback(pid, float(estimated_actual))

            # Cezayı uygula (sadece "az sattı" durumunda)
            if group_penalty < 1.0:
                existing = self._feedback_penalties.get(int(pid), 1.0)
                new_penalty = round(existing * group_penalty, 3)
                self._feedback_penalties[int(pid)] = new_penalty
                applied_penalties[pid] = new_penalty
            elif group_penalty > 1.0 and int(pid) in self._feedback_penalties:
                # Ödül: mevcut cezayı hafiflet
                existing = self._feedback_penalties[int(pid)]
                new_penalty = min(1.0, round(existing * group_penalty, 3))
                self._feedback_penalties[int(pid)] = new_penalty
                applied_penalties[pid] = new_penalty

        # Feedback geçmişine kaydet
        self.feedback_history.append({
            "type": "batch",
            "label": label,
            "period_key": period_key,
            "n_products": n_products,
            "predicted_total": total_predicted,
            "actual_total": total_sold,
            "error_rate": round(error_rate, 3),
            "group_penalty": group_penalty,
        })

        if len(self.feedback_history) % 10 == 0 and len(self.feedback_history) >= 10:
            self._update_weights()

        return {
            "label": label,
            "n_products": n_products,
            "total_predicted": total_predicted,
            "total_actual": total_sold,
            "error_rate": f"%{error_rate*100:.1f}",
            "group_penalty": group_penalty,
            "penalties_applied": len(applied_penalties),
        }

    def feedback_batch(self, label_actuals: dict, period_key: str = None) -> list:
        """
        Birden fazla etiket için aynı anda toplu feedback.

        Args:
            label_actuals: {
                "TREND":      450,   # veya {"sold": 450}
                "POTANSIYEL": 120,
                "DUSEN":      8,
            }
            period_key: None = son tahmin

        Returns:
            Her etiket için sonuç listesi

        Örnek:
            engine.feedback_batch({
                "TREND": 450,
                "POTANSIYEL": 80,
            })
        """
        results = []
        for label, value in label_actuals.items():
            # {"sold": 450} veya direkt 450 formatlarını destekle
            total_sold = value if isinstance(value, int) else value.get("sold", 0)
            result = self.feedback_by_label(
                label=label,
                total_sold=total_sold,
                period_key=period_key
            )
            results.append(result)
        return results

    def feedback_top_n(self, product_sales: dict,
                       fake_trend_threshold: int = 5) -> dict:
        """
        Kullanıcının gerçek kullanım senaryosu:

          1. engine.predict() → top 5 TREND ürünü görürsün
          2. Onları üretip test edersin
          3. Geri dönüp her birinin satışını yazarsın:

          engine.feedback_top_n({
              pid_A: 2,    # 2 sattı → SAHTE TREND → ceza
              pid_B: 45,   # 45 sattı → gerçek trend → ödül
              pid_C: 8,    # 8 sattı → sahte trend → ceza
              pid_D: 30,   # 30 sattı → iyi
              pid_E: 15,   # 15 sattı → orta
          })

        Args:
            product_sales:         {product_id: actual_sold_units}
            fake_trend_threshold:  Kaç adetten az satarsa "sahte trend"
                                   varsayılan: 5 adet

        Returns:
            Özet dict — kaç gerçek trend, kaç sahte, ne öğrendi
        """
        if not self._prediction_history:
            return {"error": "Henüz tahmin yapılmadı"}

        # Son tahminden bu ürünlerin tahmin değerlerini bul
        last_key     = max(self._prediction_history.keys())
        last_snapshot = self._prediction_history[last_key]

        # Tüm etiketlerdeki ürünleri birleştir
        all_predicted = {}
        all_cats      = {}
        for lbl, grp in last_snapshot.items():
            for pid, dem in zip(grp.get("product_ids", []),
                                grp.get("predicted_demands", [])):
                all_predicted[pid] = dem
                all_cats[pid] = lbl  # hangi etiket almıştı

        real_trends   = []
        fake_trends   = []
        results       = []

        for pid, actual_sold in product_sales.items():
            pid = int(pid)
            predicted   = all_predicted.get(pid, 0)
            prev_label  = all_cats.get(pid, "?")

            # ── Sahte Trend Tespiti ──────────────────────────────────
            # 1) Mutlak eşik: 5 adetten az sattıysa sahte trend
            # 2) Oran eşiği: tahminin %10'undan az sattıysa sahte trend
            is_fake = (
                actual_sold < fake_trend_threshold or
                (predicted > 0 and actual_sold < predicted * 0.10)
            )

            if is_fake:
                # Ciddi ceza — bu ürün sahte trend
                existing = self._feedback_penalties.get(pid, 1.0)
                new_penalty = round(existing * 0.35, 3)
                self._feedback_penalties[pid] = new_penalty
                fake_trends.append(pid)
                verdict = f"❌ SAHTE TREND (ceza: ×{new_penalty:.2f})"
            else:
                # Gerçek trend — mevcut cezayı hafiflet veya sil
                if pid in self._feedback_penalties:
                    old = self._feedback_penalties[pid]
                    # Cezayı %50 azalt
                    self._feedback_penalties[pid] = min(1.0, round(old * 1.5, 3))
                real_trends.append(pid)
                verdict = "✅ Gerçek trend"

            # Kalman'ı güncelle
            self.kalman.update_product_with_feedback(pid, float(actual_sold))

            results.append({
                "product_id":  pid,
                "prev_label":  prev_label,
                "predicted":   predicted,
                "actual_sold": actual_sold,
                "is_fake":     is_fake,
                "verdict":     verdict,
            })

            # Feedback geçmişi
            self.feedback_history.append({
                "type": "top_n",
                "product_id": pid,
                "predicted": predicted,
                "actual": actual_sold,
                "is_fake_trend": is_fake,
            })

        if len(self.feedback_history) % 10 == 0 and len(self.feedback_history) >= 10:
            self._update_weights()

        return {
            "total_tested":   len(product_sales),
            "real_trends":    len(real_trends),
            "fake_trends":    len(fake_trends),
            "fake_trend_ids": fake_trends,
            "details":        results,
        }

    def _update_weights(self):
        """[YENi] Feedback'e gore ensemble agirliklarini guncelle."""
        recent = self.feedback_history[-20:]
        if not recent:
            return

        errors = [f["error"] for f in recent]
        avg_error = np.mean(errors)

        # Hata buyukse Kalman'a daha cok agirlik ver (daha reaktif)
        if avg_error > 50:
            self.weights["catboost"] = 0.5
            self.weights["kalman_product"] = 0.3
            self.weights["kalman_category"] = 0.2
        elif avg_error > 20:
            self.weights["catboost"] = 0.55
            self.weights["kalman_product"] = 0.27
            self.weights["kalman_category"] = 0.18
        else:
            self.weights["catboost"] = 0.6
            self.weights["kalman_product"] = 0.24
            self.weights["kalman_category"] = 0.16

    def predict_for_inventory(self, material: str, color: str = None,
                               category: str = None) -> dict:
        """
        Envanter eslestirme — tum egitim feature'larini kullanarak.

        Eski sorun: Elle az feature girilince rolling_avg_* kolları boş kalıyor
                    → Model hep 0 döndürüyor
        Yeni yaklaşım: Her kategori için gerçek eğitim satırını al,
                       sadece fabric/color sütunlarını değiştir, tahmin yap.
        """
        if not self.is_trained:
            return {"error": "Motor henuz egitilmedi"}

        categories = [category] if category else self._last_training_data["category"].unique()
        results = []

        for cat in categories:
            cat_data = self._last_training_data[
                self._last_training_data["category"] == cat
            ]
            if cat_data.empty:
                continue

            # Kategorinin tipik ürününü al (favori medyanına yakın)
            median_fav = cat_data["favorite_count"].median()
            typical = cat_data.iloc[
                (cat_data["favorite_count"] - median_fav).abs().argsort()[:1]
            ]

            # CatBoost'un beklediği feature'larla tahmin sat oluştur
            raw_score = self.catboost.predict_for_representative(
                typical_row=typical.iloc[0],
                override={"fabric": material, "color": color or "unknown"}
            )

            # Composite skor → üretim adedi
            # Tipik bir kategorinin median skor'u ~5-8 arası (log scale)
            # score < 2  → talep yok (0)
            # score 2-5  → düşük (5-30 adet)
            # score 5-8  → normal (30-200 adet)
            # score 8+   → yüksek (200+ adet)
            if raw_score < 2.0:
                demand = 0
            elif raw_score < 5.0:
                demand = max(1, int((raw_score - 2.0) * 10))
            elif raw_score < 8.0:
                demand = max(10, int((raw_score - 5.0) * 55 + 30))
            else:
                demand = min(5000, int((raw_score - 8.0) * 200 + 200))

            results.append({
                "category": cat,
                "material": material,
                "color": color,
                "composite_score": round(raw_score, 3),
                "predicted_demand": demand,
                "kalman_state": self.kalman.get_state(cat),
            })

        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return {
            "recommendations": results[:5],
            "best_option": results[0] if results else None,
            "star_cluster_profile": self.clusterer.get_star_profile(),
        }

    def status(self) -> dict:
        """v2: Zengin motor durumu."""
        product_trending = self.kalman.get_product_trending()
        shap = self.catboost.get_shap_explanation(top_n=10)

        return {
            "is_trained": self.is_trained,
            "catboost_trained": self.catboost.is_trained,
            "catboost_features": self.catboost.feature_importance,
            "catboost_metrics": self.catboost.metrics,
            "shap_explanation": shap,
            "kalman_category_states": self.kalman.get_all_states(),
            "kalman_trending_products": product_trending[:10],
            "cluster_profiles": self.clusterer.cluster_profiles,
            "ensemble_weights": self.weights,
            "feedback_count": len(self.feedback_history),
            "prophet_enabled": self.use_prophet,
            "clip_enabled": self.use_clip,
        }
