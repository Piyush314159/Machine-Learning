# 🌳 Decision Tree (CART) — From Scratch

A **complete CART (Classification and Regression Tree)** classifier implemented from scratch using only NumPy — no sklearn for the tree itself. Built for deep understanding, not just usage.

## 📁 Files

| File | Description |
|------|-------------|
| `cart_from_scratch.py` | Full CART implementation — Gini impurity, best-split search, recursive tree building, prediction, and pretty-print |
| `decision_boundary_viz.py` | Decision boundary visualization across 3 datasets, depth effects, overfitting curves, and tree growth animation |

## 🧠 Concepts Covered

- **Gini Impurity** — measuring node purity
- **Best Split Search** — brute-force over every (feature, threshold) pair
- **Recursive Tree Building** — with stopping criteria (max_depth, min_samples, purity)
- **Decision Boundary Visualization** — understanding axis-aligned splits
- **Overfitting / Bias-Variance Tradeoff** — train vs test accuracy across tree depths

## 🚀 How to Run

```bash
pip install numpy matplotlib scikit-learn

# Run the CART implementation (prints tree + compares with sklearn)
python cart_from_scratch.py

# Generate 4 visualization figures
python decision_boundary_viz.py
```

## 📊 Visualizations Generated

1. **Our CART vs sklearn** — proves the scratch implementation produces identical boundaries
2. **Depth effect** — underfitting (depth=1) → balanced (depth=3) → overfitting (depth=10)
3. **Overfitting curve** — classic train vs test accuracy across depths
4. **Tree growth** — step-by-step how the tree carves up the Moons dataset

## 🔑 Key Takeaways

- Decision boundaries are always **axis-aligned** (vertical/horizontal) — fundamental limitation of single-feature splits
- **Deeper ≠ better** — test accuracy peaks around depth 3-5, then drops (overfitting)
- Our scratch tree matches sklearn because the algorithm is identical: **greedy Gini minimization**

## 🛠️ Tech Stack

- Python 3.x
- NumPy (core implementation)
- Matplotlib (visualization)
- scikit-learn (comparison & datasets only)
