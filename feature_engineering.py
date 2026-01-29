from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List
import time

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


class GAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator,
        *,
        scoring: str = "roc_auc",
        n_splits: int = 5,
        random_state: int = 42,
        population_size: int = 40,
        generations: int = 30,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.02,
        elitism: int = 2,
        min_features: int = 10,
        max_features: Optional[int] = None,
        verbose: int = 1,         # 0/1/2
        log_every: int = 1,       # 每隔多少代打印一次（你要每代输出就=1）
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_splits = n_splits
        self.random_state = random_state
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.min_features = min_features
        self.max_features = max_features
        self.verbose = verbose
        self.log_every = log_every
        self.n_jobs = n_jobs

        # learned
        self.support_mask_ = None
        self.selected_features_ = None
        self.best_score_ = None
        self.history_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = self._ensure_df(X)
        y = pd.Series(y).reset_index(drop=True)
        rng = np.random.default_rng(self.random_state)

        n_features = X.shape[1]
        max_feat = self.max_features if self.max_features is not None else n_features
        max_feat = int(np.clip(max_feat, 1, n_features))

        if self.min_features < 1:
            raise ValueError("min_features must be >= 1")
        if self.min_features > max_feat:
            raise ValueError("min_features cannot be > max_features")

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        # ---- helpers ----
        def repair(mask: np.ndarray) -> np.ndarray:
            """确保选中特征数在[min_features, max_feat]范围内"""
            mask = mask.astype(np.int8, copy=True)
            k = int(mask.sum())
            if k < self.min_features:
                zeros = np.where(mask == 0)[0]
                add = rng.choice(zeros, size=self.min_features - k, replace=False)
                mask[add] = 1
            elif k > max_feat:
                ones = np.where(mask == 1)[0]
                drop = rng.choice(ones, size=k - max_feat, replace=False)
                mask[drop] = 0
            return mask

        def init_population() -> np.ndarray:
            pop = np.zeros((self.population_size, n_features), dtype=np.int8)
            for i in range(self.population_size):
                k = int(rng.integers(self.min_features, max_feat + 1))
                idx = rng.choice(n_features, size=k, replace=False)
                pop[i, idx] = 1
                pop[i] = repair(pop[i])
            return pop

        def fitness(mask: np.ndarray) -> float:
            cols = np.where(mask == 1)[0]
            if cols.size < self.min_features:
                return -np.inf
            X_sub = X.iloc[:, cols]
            est = clone(self.estimator)
            scores = cross_val_score(
                est, X_sub, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs
            )
            return float(np.mean(scores))

        def tournament_select(pop: np.ndarray, fit: np.ndarray, t: int = 3) -> np.ndarray:
            idx = rng.choice(pop.shape[0], size=t, replace=False)
            best = idx[np.argmax(fit[idx])]
            return pop[best].copy()

        def one_point_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if n_features <= 1:
                return a.copy(), b.copy()
            p = int(rng.integers(1, n_features))
            c1 = np.concatenate([a[:p], b[p:]]).astype(np.int8)
            c2 = np.concatenate([b[:p], a[p:]]).astype(np.int8)
            return c1, c2

        def mutate(mask: np.ndarray) -> np.ndarray:
            m = mask.copy()
            flip = rng.random(n_features) < self.mutation_rate
            m[flip] = 1 - m[flip]
            return repair(m)

        # ---- GA loop ----
        t0 = time.perf_counter()

        pop = init_population()
        fit_vals = np.array([fitness(ind) for ind in pop], dtype=float)

        history: List[Dict[str, Any]] = []

        for gen in range(self.generations):
            # elitism
            elite_idx = np.argsort(fit_vals)[::-1][: max(0, self.elitism)]
            elites = pop[elite_idx].copy()

            new_pop = []
            while len(new_pop) < self.population_size - elites.shape[0]:
                p1 = tournament_select(pop, fit_vals)
                p2 = tournament_select(pop, fit_vals)

                if rng.random() < self.crossover_rate:
                    c1, c2 = one_point_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = mutate(repair(c1))
                c2 = mutate(repair(c2))

                new_pop.append(c1)
                if len(new_pop) < self.population_size - elites.shape[0]:
                    new_pop.append(c2)

            pop = np.vstack([elites, np.array(new_pop, dtype=np.int8)])
            fit_vals = np.array([fitness(ind) for ind in pop], dtype=float)

            best_i = int(np.argmax(fit_vals))
            best_score = float(fit_vals[best_i])
            best_k = int(pop[best_i].sum())

            elapsed = time.perf_counter() - t0
            mean_score = float(np.mean(fit_vals))
            std_score = float(np.std(fit_vals))

            history.append(
                {
                    "gen": gen,
                    "best_score": best_score,
                    "best_k": best_k,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "elapsed_sec": elapsed,
                }
            )

            # ---- per-generation output ----
            if self.verbose and (gen % max(1, int(self.log_every)) == 0):
                if self.verbose >= 2:
                    print(
                        f"[GA] gen={gen:02d} "
                        f"best={best_score:.6f} k={best_k} "
                        f"mean={mean_score:.6f} std={std_score:.6f} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[GA] gen={gen:02d} best_score={best_score:.6f} best_k={best_k}",
                        flush=True,
                    )

        best_i = int(np.argmax(fit_vals))
        best_mask = pop[best_i].astype(bool)
        self.support_mask_ = best_mask
        self.selected_features_ = X.columns[best_mask].tolist()
        self.best_score_ = float(fit_vals[best_i])
        self.history_ = history
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._ensure_df(X)
        if self.support_mask_ is None:
            raise RuntimeError("GAFeatureSelector is not fitted yet.")
        return X.loc[:, self.selected_features_]

    def get_support(self) -> np.ndarray:
        if self.support_mask_ is None:
            raise RuntimeError("GAFeatureSelector is not fitted yet.")
        return self.support_mask_

    @staticmethod
    def _ensure_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
