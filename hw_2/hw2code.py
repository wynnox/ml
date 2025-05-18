import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    order = np.argsort(feature_vector)
    x_sorted = feature_vector[order]
    y_sorted = target_vector[order]

    midpoints = (x_sorted[1:] + x_sorted[:-1]) / 2
    valid = x_sorted[1:] != x_sorted[:-1]
    thresholds = midpoints[valid]

    if thresholds.size == 0:
        return np.array([]), np.array([]), None, None

    total = len(y_sorted)
    total_pos = np.sum(y_sorted)

    left_pos = np.cumsum(y_sorted)[:-1][valid]
    left_count = np.arange(1, total)[valid]
    right_count = total - left_count
    right_pos = total_pos - left_pos

    p1_l = left_pos / left_count
    p1_r = right_pos / right_count

    gini_left = 1 - p1_l**2 - (1 - p1_l)**2
    gini_right = 1 - p1_r**2 - (1 - p1_r)**2

    gini_total = -(left_count / total) * gini_left - (right_count / total) * gini_right

    best_idx = np.argmin(gini_total)

    return thresholds, gini_total, thresholds[best_idx], gini_total[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, float("inf"), None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: clicks.get(key, 0) / counts[key] if counts[key] > 0 else 0 for key in counts}
                sorted_categories = [cat for cat, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map.get(x, 0) for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            temp_split = feature_vector < threshold
            left_size = np.sum(temp_split)
            right_size = len(temp_split) - left_size

            if self._min_samples_leaf is not None and (
                    left_size < self._min_samples_leaf or right_size < self._min_samples_leaf
            ):
                continue

            if gini < gini_best:
                feature_best = feature
                gini_best = gini
                split = temp_split
                if feature_type == "real":
                    threshold_best = threshold
                else:
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}

        feature_type = self._feature_types[feature_best]
        feature_vector = sub_X[:, feature_best]

        if feature_type == "real":
            split = feature_vector.astype(float) < threshold_best
        else:
            split = np.isin(feature_vector, threshold_best)

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if float(x[feature]) < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._tree = {}
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])