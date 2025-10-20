import numpy as np
from dataclasses import dataclass

@dataclass
class UpliftTreeRegressor:
    max_depth: int = 3
    min_samples_leaf: int = 1000
    min_samples_leaf_treated: int = 300
    min_samples_leaf_control: int = 300

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """fit"""
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, treatment, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict"""
        preds = np.array([self._predict_row(x, self.tree_) for x in X])
        return preds


    def _build_tree(self, X, treatment, y, depth):
        """building recoursively"""
        n_samples = len(y)
        if (depth >= self.max_depth) or (n_samples < self.min_samples_leaf):
            return self._create_leaf(treatment, y)

        best_gain = 0.0
        best_split = None

        for feature_idx in range(self.n_features_):
            column = X[:, feature_idx]
            thresholds = self._get_thresholds(column)

            for thr in thresholds:
                left_mask = column <= thr
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                # Проверяем доли treatment/control
                if (sum(treatment[left_mask] == 1) < self.min_samples_leaf_treated or
                    sum(treatment[left_mask] == 0) < self.min_samples_leaf_control or
                    sum(treatment[right_mask] == 1) < self.min_samples_leaf_treated or
                    sum(treatment[right_mask] == 0) < self.min_samples_leaf_control):
                    continue

                gain = self._calc_uplift_gain(y, treatment, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, thr, left_mask, right_mask)

        if best_split is None:
            return self._create_leaf(treatment, y)

        f_idx, thr, left_mask, right_mask = best_split
        left = self._build_tree(X[left_mask], treatment[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], treatment[right_mask], y[right_mask], depth + 1)

        return {
            "feature": f_idx,
            "threshold": thr,
            "left": left,
            "right": right,
        }

    def _create_leaf(self, treatment, y):
        """leaves creation"""
        treated_mask = treatment == 1
        control_mask = treatment == 0

        # check for 0 observations
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            uplift = 0.0
        else:
            uplift = y[treated_mask].mean() - y[control_mask].mean()

        return {"uplift": uplift}

    def _get_thresholds(self, column_values):
        """Thresholds (examples)"""
        unique_values = np.unique(column_values)
        if len(unique_values) > 10:
            percentiles = np.percentile(
                column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]
            )
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])
        threshold_options = np.unique(percentiles)
        return threshold_options

    def _calc_uplift_gain(self, y, treatment, left_mask, right_mask):
        """Def uplift between leaves"""
        def uplift(y, t):
            treated = y[t == 1]
            control = y[t == 0]
            if len(treated) == 0 or len(control) == 0:
                return 0.0
            return treated.mean() - control.mean()

        left_u = uplift(y[left_mask], treatment[left_mask])
        right_u = uplift(y[right_mask], treatment[right_mask])

        # Gain
        gain = (len(y[left_mask]) * left_u**2 + len(y[right_mask]) * right_u**2) / len(y)
        return gain

    def _predict_row(self, x, node):
        """"""
        if "uplift" in node:
            return node["uplift"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_row(x, node["left"])
        else:
            return self._predict_row(x, node["right"])
