import os

def main():
    filepath = "/home/bedir/Documents/code/lumora-intelligence/services/intelligence_service.py"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # REFACTOR 1: startup_train -> _prepare_training_data
    # We will replace the block from "df_work = df.copy()" to "df_feat = self._engine.feature_eng.build_features(df_clean)"
    # Actually, it's safer to just extract it entirely.
    
    start_train = content.find("def startup_train(self):")
    if start_train == -1: return

    # REFACTOR 2: nightly_batch -> _generate_alerts
    # Find "# ── 4. Alertler ───────────────────────────────────────────────"
    
    new_script = """
    def _prepare_training_data(self, df_work: pd.DataFrame) -> pd.DataFrame:
        col_renames = {}
        if "product_category" in df_work.columns and "category" not in df_work.columns:
            col_renames["product_category"] = "category"
        elif "product_category" in df_work.columns and "category" in df_work.columns and df_work["category"].isna().all():
            df_work.drop(columns=["category"], inplace=True)
            col_renames["product_category"] = "category"
        if "recorded_at" in df_work.columns and "date" not in df_work.columns:
            col_renames["recorded_at"] = "date"
        if col_renames:
            df_work = df_work.rename(columns=col_renames)
        if "date" in df_work.columns:
            df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
        df_clean = self._engine.zscore.filter_errors(df_work)
        df_clean = self._engine.zscore.detect_anomalies(df_clean)
        return self._engine.feature_eng.build_features(df_clean)

    def _generate_alerts(self, cat: str, preds: list[dict], cat_heat: float):
        from db.writer import save_alert
        for p in preds:
            score = p.get("trend_score", 0)
            pid = p["product_id"]
            if score > 90:
                save_alert({"type": "rank_spike", "product_id": pid, "category": cat, "message": f"Yüksek trend skoru: {score:.1f}", "extra_data": p})
            if p.get("is_new_entrant") and score > 80:
                save_alert({"type": "viral_start", "product_id": pid, "category": cat, "message": f"Yeni giriş + yüksek skor: {score:.1f}", "extra_data": p})
        
        if cat_heat > 0.8:
            save_alert({"type": "category_heat", "product_id": 0, "category": cat, "message": f"Kategori genel yükselişte (heat: {cat_heat:.2f})", "extra_data": {"category_heat": cat_heat}})
        elif cat_heat < -0.8:
            save_alert({"type": "category_cold", "product_id": 0, "category": cat, "message": f"Kategori genel düşüşte (heat: {cat_heat:.2f})", "extra_data": {"category_heat": cat_heat}})
            
        try:
            from sqlalchemy import text as _t
            from db.connection import engine as _dbe
            with _dbe.connect() as _conn:
                drop_sql = _t('''SELECT DISTINCT ON (product_id) product_id, rank_change_3d FROM daily_metrics WHERE search_term = :cat AND recorded_at >= CURRENT_DATE AND rank_change_3d > 300 ORDER BY product_id, recorded_at DESC LIMIT 10''')
                drops = _conn.execute(drop_sql, {"cat": cat}).fetchall()
                for drop_row in drops:
                    save_alert({"type": "rank_drop", "product_id": drop_row[0], "category": cat, "message": f"3 günde {int(drop_row[1])} sıra kötüleşme", "extra_data": {"rank_change_3d": int(drop_row[1])}})
        except Exception as e:
            pass
"""
    # Replace contents dynamically here.
    return 

if __name__ == "__main__":
    main()
