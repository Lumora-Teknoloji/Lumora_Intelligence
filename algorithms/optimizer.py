# algorithms/optimizer.py
"""
Algoritma 8: Bayesian Optimization (Optuna)
CatBoost hyperparameter otomatik optimizasyonu.
"""
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import cross_val_score
import numpy as np
from config import OPTUNA_N_TRIALS

# Optuna loglarını sustur
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Bayesian Optimization ile CatBoost'un optimal parametrelerini bulur.

    Grid Search: tüm kombinasyonları dene (yavaş, verimsiz)
    Bayesian: önceki denemelere bakarak akıllı arama (hızlı, verimli)

    50-100 deneme ile binlerce kombinasyonu taramış gibi sonuç üretir.
    """

    def __init__(self, n_trials=None):
        self.n_trials = n_trials or OPTUNA_N_TRIALS
        self.best_params = None
        self.study = None

    def optimize(self, X, y, cat_indices=None, verbose=True) -> dict:
        """
        CatBoost için optimal hyperparameter bulur.

        Args:
            X: Feature matrix (DataFrame veya numpy)
            y: Target (Series veya numpy)
            cat_indices: Kategorik sütun indeksleri

        Returns:
            best_params: Optimal parametreler
        """
        if len(X) < 10:
            print("  ⚠ Optimizer: Yeterli veri yok, varsayılan parametreler kullanılacak")
            self.best_params = self._default_params()
            return self.best_params

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, max(2, len(X) // 10)),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            }

            model = CatBoostRegressor(
                **params,
                cat_features=cat_indices,
                verbose=0,
                random_seed=42,
                early_stopping_rounds=30,
            )

            # 3-fold cross validation
            n_splits = min(3, len(X))
            if n_splits < 2:
                return float("inf")

            try:
                # Manuel CV (CatBoost uyumlu)
                fold_size = len(X) // n_splits
                scores = []

                for i in range(n_splits):
                    val_start = i * fold_size
                    val_end = val_start + fold_size

                    X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0) if isinstance(X, np.ndarray) else X.iloc[list(range(val_start)) + list(range(val_end, len(X)))]
                    y_train = np.concatenate([y[:val_start], y[val_end:]]) if isinstance(y, np.ndarray) else y.iloc[list(range(val_start)) + list(range(val_end, len(y)))]
                    X_val = X[val_start:val_end] if isinstance(X, np.ndarray) else X.iloc[val_start:val_end]
                    y_val = y[val_start:val_end] if isinstance(y, np.ndarray) else y.iloc[val_start:val_end]

                    if len(X_train) < 3 or len(X_val) < 1:
                        continue

                    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
                    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

                    model.fit(train_pool, eval_set=val_pool, verbose=0)
                    preds = model.predict(X_val)
                    mae = np.mean(np.abs(preds - np.array(y_val, dtype=float)))
                    scores.append(mae)

                return np.mean(scores) if scores else float("inf")

            except Exception:
                return float("inf")

        if verbose:
            print(f"  🔍 Bayesian Optimization başlıyor ({self.n_trials} deneme)...")

        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = self.study.best_params

        if verbose:
            print(f"  ✓ En iyi parametreler bulundu (MAE: {self.study.best_value:.2f})")
            for k, v in self.best_params.items():
                print(f"    {k}: {v}")

        return self.best_params

    def get_optimized_model(self, cat_indices=None) -> CatBoostRegressor:
        """Optimize edilmiş parametrelerle CatBoost modeli döner."""
        params = self.best_params or self._default_params()
        return CatBoostRegressor(
            **params,
            cat_features=cat_indices,
            verbose=0,
            random_seed=42,
        )

    def _default_params(self):
        """Varsayılan parametreler."""
        return {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "min_data_in_leaf": 5,
            "random_strength": 1.0,
        }
