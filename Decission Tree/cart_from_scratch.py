"""
=============================================================================
CART Decision Tree — Built from Scratch (Numeric Features Only)
=============================================================================

Goal: Understand **every moving part** of a Classification And Regression Tree.
      We only use numpy; no sklearn for the tree itself.

Key concepts covered (in order):
  1. Gini Impurity — how we measure "purity" of a node
  2. Finding the best split — brute-force over every feature × threshold
  3. Recursive tree building — when to stop, when to split
  4. Prediction — walking a sample down the tree
  5. Accuracy check — compare with sklearn's DecisionTreeClassifier

Run:  python cart_from_scratch.py
=============================================================================
"""

import numpy as np
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# 1. GINI IMPURITY
# ─────────────────────────────────────────────────────────────────────────────
# Gini tells us "if I randomly pick two samples from this node and label one
# with the other's label, how often would I be wrong?"
#
#   Gini(node) = 1 − Σ (pᵢ)²     where pᵢ = fraction of class i in the node
#
# Pure node  →  Gini = 0   (all samples same class)
# Worst case →  Gini = 0.5  (binary, 50-50 split)

def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for a set of labels y."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)               # [count_class_0, count_class_1, ...]
    probabilities = counts / len(y)        # convert to fractions
    return 1.0 - np.sum(probabilities ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEIGHTED GINI FOR A SPLIT
# ─────────────────────────────────────────────────────────────────────────────
# When we split a node into left & right children, the "quality" of that split
# is the **weighted average** of the children's Gini values:
#
#   Gini_split = (n_left/n) * Gini(left) + (n_right/n) * Gini(right)
#
# We want to **minimise** this.

def weighted_gini(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Weighted Gini impurity across a left-right split."""
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * gini_impurity(y_left) + \
           (len(y_right) / n) * gini_impurity(y_right)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIND THE BEST SPLIT — THE HEART OF CART
# ─────────────────────────────────────────────────────────────────────────────
# For each feature:
#   - Sort the unique values
#   - Try every midpoint between consecutive unique values as a threshold
#   - Compute the weighted Gini for that split
#   - Keep the (feature, threshold) pair that gives the **lowest** Gini
#
# This is O(n × m × log n) per node — brute force but simple & correct.

def find_best_split(X: np.ndarray, y: np.ndarray):
    """
    Returns (best_feature_index, best_threshold, best_gini).
    Returns (None, None, None) if no valid split exists.
    """
    n_samples, n_features = X.shape
    best_gini = float('inf')
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        # Get sorted unique values for this feature
        values = np.sort(np.unique(X[:, feature_idx]))

        # Try midpoints between consecutive unique values
        # (more robust than trying every raw value)
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            # Split: left = samples where feature <= threshold
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            # Skip if split produces an empty child
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gini = weighted_gini(y_left, y_right)

            if gini < best_gini:
                best_gini = gini
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gini


# ─────────────────────────────────────────────────────────────────────────────
# 4. THE TREE NODE
# ─────────────────────────────────────────────────────────────────────────────
# Each node is either:
#   - A LEAF: stores the predicted class (majority vote)
#   - An INTERNAL node: stores (feature_index, threshold, left_child, right_child)

class Node:
    def __init__(self, *, feature=None, threshold=None,
                 left=None, right=None, predicted_class=None,
                 gini=None, n_samples=None, class_counts=None):
        # Internal node attributes
        self.feature = feature            # which feature to split on
        self.threshold = threshold        # split value
        self.left = left                  # left child Node
        self.right = right                # right child Node

        # Leaf / diagnostic attributes
        self.predicted_class = predicted_class  # majority class at this node
        self.gini = gini
        self.n_samples = n_samples
        self.class_counts = class_counts        # e.g. {0: 30, 1: 20}

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. DECISION TREE CLASSIFIER (CART)
# ─────────────────────────────────────────────────────────────────────────────

class DecisionTreeCART:
    """
    A from-scratch CART classifier for numeric features.

    Hyperparameters (stopping criteria):
      - max_depth:        how deep the tree can grow
      - min_samples_split: minimum samples required to attempt a split
      - min_samples_leaf:  minimum samples required in each child
    """

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes_ = None
        self.feature_names_ = None

    # ── Fit ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        """Build the tree recursively."""
        self.n_classes_ = len(np.unique(y))
        self.feature_names_ = feature_names
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree."""
        n_samples = len(y)
        class_counts = dict(Counter(y))
        majority_class = max(class_counts, key=class_counts.get)
        current_gini = gini_impurity(y)

        # ── Create a node with diagnostic info ──
        node = Node(
            predicted_class=majority_class,
            gini=current_gini,
            n_samples=n_samples,
            class_counts=class_counts,
        )

        # ── STOPPING CONDITIONS ──────────────────────────────────────────
        # 1. Reached max depth
        if depth >= self.max_depth:
            return node

        # 2. Node is already pure (Gini = 0)
        if current_gini == 0.0:
            return node

        # 3. Not enough samples to split
        if n_samples < self.min_samples_split:
            return node

        # ── TRY TO SPLIT ─────────────────────────────────────────────────
        best_feat, best_thresh, best_gini = find_best_split(X, y)

        # No valid split found
        if best_feat is None:
            return node

        # Split the data
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # 4. Check min_samples_leaf constraint
        if np.sum(left_mask) < self.min_samples_leaf or \
           np.sum(right_mask) < self.min_samples_leaf:
            return node

        # ── RECURSE ──────────────────────────────────────────────────────
        node.feature = best_feat
        node.threshold = best_thresh
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    # ── Predict ──────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each sample in X."""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x: np.ndarray, node: Node) -> int:
        """Walk one sample down the tree until we hit a leaf."""
        if node.is_leaf:
            return node.predicted_class

        # Go left if feature value <= threshold, else go right
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    # ── Pretty-print the tree ────────────────────────────────────────────
    def print_tree(self, node=None, indent="", prefix="ROOT"):
        """Recursively print the tree structure (great for debugging)."""
        if node is None:
            node = self.root

        # Feature name
        feat_name = (self.feature_names_[node.feature]
                     if self.feature_names_ and node.feature is not None
                     else f"X[{node.feature}]")

        if node.is_leaf:
            print(f"{indent}{prefix} → LEAF  class={node.predicted_class}  "
                  f"gini={node.gini:.3f}  n={node.n_samples}  "
                  f"counts={node.class_counts}")
        else:
            print(f"{indent}{prefix}  {feat_name} <= {node.threshold:.4f}?  "
                  f"gini={node.gini:.3f}  n={node.n_samples}  "
                  f"counts={node.class_counts}")
            self.print_tree(node.left, indent + "  │ ", prefix="L")
            self.print_tree(node.right, indent + "  │ ", prefix="R")


# ═════════════════════════════════════════════════════════════════════════════
# DEMO: Train on the Iris dataset and compare with sklearn
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier as SklearnDT

    print("=" * 70)
    print("  CART Decision Tree — From Scratch vs sklearn")
    print("=" * 70)

    # Load Iris (4 numeric features, 3 classes)
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ── Our scratch-built tree ───────────────────────────────────────────
    my_tree = DecisionTreeCART(max_depth=5, min_samples_split=2, min_samples_leaf=1)
    my_tree.fit(X_train, y_train, feature_names=list(feature_names))

    print("\n📊 Our scratch-built tree structure:\n")
    my_tree.print_tree()

    my_preds = my_tree.predict(X_test)
    my_acc = np.mean(my_preds == y_test)

    # ── sklearn's tree (same hyperparams) ────────────────────────────────
    sk_tree = SklearnDT(max_depth=5, min_samples_split=2, min_samples_leaf=1,
                        criterion="gini", random_state=42)
    sk_tree.fit(X_train, y_train)
    sk_preds = sk_tree.predict(X_test)
    sk_acc = np.mean(sk_preds == y_test)

    # ── Results ──────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print(f"  Our tree accuracy:     {my_acc:.4f}  ({int(my_acc * len(y_test))}/{len(y_test)})")
    print(f"  sklearn tree accuracy: {sk_acc:.4f}  ({int(sk_acc * len(y_test))}/{len(y_test)})")
    print("─" * 50)

    # Show where they disagree (if anywhere)
    disagreements = np.where(my_preds != sk_preds)[0]
    if len(disagreements) == 0:
        print("  ✅ Both trees agree on every test sample!")
    else:
        print(f"  ⚠️  Disagree on {len(disagreements)} samples: {disagreements}")
        for idx in disagreements[:5]:
            print(f"     Sample {idx}: ours={my_preds[idx]}, sklearn={sk_preds[idx]}, "
                  f"true={y_test[idx]}")

    print("\n💡 Key takeaway: Our from-scratch CART matches or nearly matches")
    print("   sklearn because the algorithm is identical — greedy Gini splits.")
    print("   Minor differences come from tie-breaking when two splits have")
    print("   the same Gini.\n")
