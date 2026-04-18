import re

def main():
    filepath = "/home/bedir/Documents/code/lumora-intelligence/engine/predictor.py"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1) Locate start of def predict
    predict_start = content.find("    def predict(self, df: pd.DataFrame = None, tuning_params: dict = None,")
    # 2) Locate start of def feedback
    feedback_start = content.find("    def feedback(self, category: str, actual_sales: int, predicted_demand: int = None,")
    
    if predict_start == -1 or feedback_start == -1:
        print("Could not find predict or feedback methods.")
        return
        
    before_predict = content[:predict_start]
    after_feedback = content[feedback_start:]
    
    # The new refactored block
    new_predict_block = """    def _resolve_params(self, category, tuning_params, category_tuning):
        tp = {}
        for key, default in self.DEFAULT_TUNING.items():
            if category_tuning and category in category_tuning and key in category_tuning[category]:
                tp[key] = category_tuning[category][key]
            elif tuning_params and key in tuning_params:
                tp[key] = tuning_params[key]
            else:
                tp[key] = default
        return tp

    def _compute_growth_rates(self, data, use_cache):
        if use_cache and getattr(self, "_cached_growth_df", None) is not None:
            return self._cached_growth_df
            
        growth_results = []
        for pid, pdata in data.groupby("product_id"):
            pdata = pdata.sort_values("date")
            n = len(pdata)
            if n < 6:
                growth_results.append({"product_id": pid, "cart_growth_pct": 0.0})
                continue
            first = pdata.head(max(3, n // 3))["cart_count"].mean()
            last = pdata.tail(max(3, n // 3))["cart_count"].mean()
            growth = (last - first) / (first + 1e-6) * 100
            growth_results.append({"product_id": pid, "cart_growth_pct": round(growth, 1)})
            
        growth_df = pd.DataFrame(growth_results)
        if use_cache:
            self._cached_growth_df = growth_df
        return growth_df

    def _apply_scoring(self, result, data, tuning_params, category_tuning):
        def norm(s):
            r = s.max() - s.min()
            if r < 1e-6:
                return pd.Series(0.5, index=s.index)
            return (s - s.min()) / r

        scored_parts = []
        for cat, cat_result in result.groupby("category"):
            cat_result = cat_result.copy()
            tp = self._resolve_params(cat, tuning_params, category_tuning)
            cr = tp["clip_ranges"]

            vel = cat_result["kalman_prod_velocity"].fillna(0)
            gr  = cat_result["cart_growth_pct"].fillna(0)
            dem = cat_result["predicted_demand"].fillna(0)

            score_g = norm(gr.clip(lower=cr["growth"][0], upper=cr["growth"][1]))
            score_v = norm(vel.clip(lower=cr["velocity"][0], upper=cr["velocity"][1]))

            if "predicted_demand" in cat_result.columns:
                score_cb = norm(cat_result["predicted_demand"].fillna(0).clip(lower=0))
            else:
                score_cb = pd.Series(0.5, index=cat_result.index)

            if "favorite_growth_14d" in cat_result.columns:
                score_fav14 = norm(cat_result["favorite_growth_14d"].fillna(1.0).clip(0.05, 10.0))
            else:
                score_fav14 = pd.Series(0.5, index=cat_result.index)

            if "favorite_growth_3d" in cat_result.columns:
                score_fav3 = norm(cat_result["favorite_growth_3d"].fillna(1.0).clip(0.05, 15.0))
            else:
                score_fav3 = score_fav14

            score_fav = score_fav3 * 0.60 + score_fav14 * 0.40

            data_cat = data[data["category"] == cat]
            if "rank_reach_mult" in data_cat.columns:
                reach_map = data_cat.sort_values("date").groupby("product_id")["rank_reach_mult"].last().to_dict()
                score_reach = norm(cat_result["product_id"].map(reach_map).fillna(0.1))
            else:
                score_reach = pd.Series(0.5, index=cat_result.index)

            cat_result["trend_score"] = (
                score_reach * 0.30 + score_cb * 0.30 + score_v * 0.20 +
                score_fav * 0.15 + score_g * 0.05
            ) * 100
            cat_result["trend_score"] = cat_result["trend_score"].round(1)

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

            scores = cat_result["trend_score"]
            if nc >= 6:
                p85, p55, p25 = max(scores.quantile(0.85), 62.0), max(scores.quantile(0.55), 42.0), max(scores.quantile(0.25), 22.0)
            elif nc >= 4:
                p85, p55, p25 = max(scores.quantile(0.80), 60.0), max(scores.quantile(0.50), 40.0), max(scores.quantile(0.25), 20.0)
            else:
                p85, p55, p25 = 70.0, 45.0, 25.0

            cat_result["trend_label"] = np.where(
                scores >= p85, "TREND",
                np.where(scores >= p55, "POTANSIYEL",
                np.where(scores >= p25, "STABIL", "DUSEN")))

            scored_parts.append(cat_result)

        return pd.concat(scored_parts, ignore_index=True)

    def _apply_overrides(self, result):
        if "abs_rank_change_7d" in result.columns and "favorite_growth_14d" in result.columns:
            dusen_override = (result["abs_rank_change_7d"] > 300) & (result["favorite_growth_14d"] < 0.80)
            result.loc[dusen_override, "trend_label"] = "DUSEN"

        if "is_fav_spike" in result.columns and "rank_improving_strong" in result.columns:
            spike_mask = (result["is_fav_spike"] == 1) & (result["rank_improving_strong"] == 1)
            result.loc[spike_mask, "trend_label"] = "TREND"
        return result

    def _apply_penalties(self, result):
        if self._feedback_penalties:
            result["feedback_penalty"] = result["product_id"].map(lambda pid: self._feedback_penalties.get(int(pid), 1.0))
            result["trend_score"] = (result["trend_score"] * result["feedback_penalty"]).round(1)
            result["ensemble_demand"] = (result["ensemble_demand"] * result["feedback_penalty"]).clip(lower=0).round(0).astype(int)

            lt = self.DEFAULT_TUNING["label_thresholds"]
            result["trend_label"] = np.where(
                result["trend_score"] >= lt["trend"], "TREND",
                np.where(result["trend_score"] >= lt["potansiyel"], "POTANSIYEL",
                np.where(result["trend_score"] >= lt["stabil"], "STABIL", "DUSEN")))
        return result

    def predict(self, df: pd.DataFrame = None, tuning_params: dict = None,
                category_tuning: dict = None) -> pd.DataFrame:
        if not self.is_trained:
            print("Motor henuz egitilmedi!")
            return pd.DataFrame()

        self._decay_penalties()

        use_cache = (df is None or df is getattr(self, "_last_training_data", None))
        data = getattr(self, "_last_training_data", pd.DataFrame()) if df is None else df

        if use_cache and getattr(self, "_cached_cb_preds", None) is not None:
            cb_predictions = self._cached_cb_preds
        else:
            all_cb_predictions = []
            for cat, cat_data in data.groupby("category"):
                cat_pred = self.registry.predict_category(cat, cat_data)
                if cat_pred.empty:
                    cat_pred = self.catboost.predict(cat_data)
                if not cat_pred.empty:
                    all_cb_predictions.append(cat_pred)

            if not all_cb_predictions:
                import logging as _log
                _log.getLogger(__name__).info("CatBoost modeli yok — cold-start heuristic kullanılıyor")
                latest = data.sort_values("date").groupby("product_id").last().reset_index()
                heuristic_cols = ["product_id"]
                if "category" in latest.columns: heuristic_cols.append("category")
                heuristic = latest[heuristic_cols].copy()
                heuristic["predicted_demand"] = (
                    latest["cart_count"].fillna(0) * 0.3 + latest["favorite_count"].fillna(0) * 0.2 +
                    latest.get("engagement_score", pd.Series(0)).fillna(0) * 50
                ).clip(lower=0).round(0).astype(int)
                cb_predictions = heuristic[["product_id", "predicted_demand"]]
            else:
                cb_predictions = pd.concat(all_cb_predictions, ignore_index=True)
            if use_cache: self._cached_cb_preds = cb_predictions

        kalman_cat_df = pd.DataFrame([{"category": cat, "kalman_cat_demand": self.kalman.get_state(cat)["demand"], "kalman_cat_velocity": self.kalman.get_state(cat)["velocity"]} for cat in data["category"].unique()])
        kalman_prod_df = pd.DataFrame([{"product_id": pid, "kalman_prod_demand": self.kalman.get_product_state(pid)["demand"], "kalman_prod_velocity": self.kalman.get_product_state(pid)["velocity"]} for pid in data["product_id"].unique()])
        
        growth_df = self._compute_growth_rates(data, use_cache)

        product_cats = data.groupby("product_id")["category"].first().reset_index()
        result = cb_predictions.merge(product_cats, on="product_id", how="left")
        result = result.merge(kalman_cat_df, on="category", how="left")
        result = result.merge(kalman_prod_df, on="product_id", how="left")
        result = result.merge(growth_df, on="product_id", how="left")

        result = self._apply_scoring(result, data, tuning_params, category_tuning)
        result = self._apply_overrides(result)

        w = self.weights
        cb_d = result["predicted_demand"]
        kp_d = result["kalman_prod_demand"].fillna(cb_d)
        kc_d = result["kalman_cat_demand"].fillna(cb_d)
        result["ensemble_demand"] = np.round(cb_d * w["catboost"] + kp_d * w["kalman_product"] + kc_d * w["kalman_category"]).clip(lower=0).astype(int)

        result = self._apply_penalties(result)

        columns = ["product_id", "category", "trend_score", "trend_label", "confidence", "cart_growth_pct", "kalman_prod_velocity", "predicted_demand", "ensemble_demand", "kalman_prod_demand", "kalman_cat_demand"]
        final = result[[c for c in columns if c in result.columns]].sort_values("trend_score", ascending=False)

        self._predict_call_count += 1
        period_key = f"period_{self._predict_call_count}"
        self._prediction_history[period_key] = {lbl: {"product_ids": [int(p) for p in final[final["trend_label"] == lbl]["product_id"].tolist()], "predicted_demands": [int(d) for d in final[final["trend_label"] == lbl]["ensemble_demand"].tolist()], "total_predicted": int(sum(final[final["trend_label"] == lbl]["ensemble_demand"].tolist()))} for lbl in ["TREND", "POTANSIYEL", "STABIL", "DUSEN"]}
        
        if len(self._prediction_history) > 12:
            del self._prediction_history[min(self._prediction_history.keys())]

        return final

"""
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(before_predict + new_predict_block + after_feedback)

if __name__ == "__main__":
    main()
