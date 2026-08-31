"""
=============================================================================
Decision Boundary Visualization — See What the Tree Actually Learned
=============================================================================

This script:
  1. Uses the CART tree we built from scratch (cart_from_scratch.py)
  2. Trains it on a 2D dataset (so we can plot it!)
  3. Paints the decision regions on a mesh grid
  4. Overlays the training points
  5. Shows side-by-side comparison with sklearn's tree

Datasets available:
  - Iris (first 2 features, 3 classes)     ← default
  - make_moons (non-linear, 2 classes)
  - make_circles (concentric, 2 classes)

Run:  python decision_boundary_viz.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, make_moons, make_circles
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.model_selection import train_test_split

# Import our scratch-built tree
from cart_from_scratch import DecisionTreeCART


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Plot decision boundary for ANY classifier with a .predict() method
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_boundary(ax, clf, X, y, title, resolution=300):
    """
    How this works:
      1. Create a fine mesh grid covering the feature space
      2. Predict the class for EVERY point on the grid
      3. Color each grid point by its predicted class → gives us the "regions"
      4. Scatter the actual data points on top

    This is the standard way to visualize any classifier's decision surface.
    """

    # Color maps
    # Light pastel colors for the background regions
    cmap_background = ListedColormap(['#FFDDD2', '#C8E6C9', '#BBDEFB', '#FFF9C4'])
    # Bold colors for the data points
    cmap_points = ListedColormap(['#D84315', '#2E7D32', '#1565C0', '#F9A825'])

    # ── Step 1: Define the mesh grid ─────────────────────────────────────
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # ── Step 2: Predict class for every grid point ───────────────────────
    # Flatten the grid into (N, 2) array, predict, reshape back
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # ── Step 3: Paint the regions ────────────────────────────────────────
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_background)

    # Draw the decision boundaries as contour lines
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.3)

    # ── Step 4: Scatter the data points ──────────────────────────────────
    scatter = ax.scatter(
        X[:, 0], X[:, 1],
        c=y, cmap=cmap_points,
        edgecolors='black', linewidths=0.8, s=50, alpha=0.9,
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.tick_params(labelsize=9)

    return scatter


# ─────────────────────────────────────────────────────────────────────────────
# MAIN: Run the visualization for multiple datasets × depths
# ─────────────────────────────────────────────────────────────────────────────

def run_visualization():
    # ── Prepare 3 datasets ───────────────────────────────────────────────
    iris = load_iris()
    X_iris = iris.data[:, :2]  # Only first 2 features for 2D plotting
    y_iris = iris.target

    X_moons, y_moons = make_moons(n_samples=300, noise=0.25, random_state=42)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.15,
                                         factor=0.5, random_state=42)

    datasets = [
        ("Iris (2 features, 3 classes)", X_iris, y_iris),
        ("Moons (non-linear, 2 classes)", X_moons, y_moons),
        ("Circles (concentric, 2 classes)", X_circles, y_circles),
    ]

    depths_to_show = [1, 3, 10]  # Shallow → medium → deep

    # ═════════════════════════════════════════════════════════════════════
    # FIGURE 1: Our tree vs sklearn (depth=5) across all 3 datasets
    # ═════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 18))
    fig1.suptitle("Decision Boundaries: Our CART vs sklearn\n(max_depth=5)",
                  fontsize=16, fontweight='bold', y=0.98)

    for row, (name, X, y) in enumerate(datasets):
        # Our tree
        our_tree = DecisionTreeCART(max_depth=5)
        our_tree.fit(X, y)
        our_acc = np.mean(our_tree.predict(X) == y)
        plot_decision_boundary(
            axes1[row, 0], our_tree, X, y,
            f"Our CART — {name}\nTrain acc: {our_acc:.3f}"
        )

        # sklearn's tree
        sk_tree = SklearnDT(max_depth=5, criterion='gini', random_state=42)
        sk_tree.fit(X, y)
        sk_acc = np.mean(sk_tree.predict(X) == y)
        plot_decision_boundary(
            axes1[row, 1], sk_tree, X, y,
            f"sklearn DT — {name}\nTrain acc: {sk_acc:.3f}"
        )

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig("fig1_ours_vs_sklearn.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: fig1_ours_vs_sklearn.png")

    # ═════════════════════════════════════════════════════════════════════
    # FIGURE 2: Effect of max_depth (overfitting visualization)
    # ═════════════════════════════════════════════════════════════════════
    # This is crucial for understanding WHEN a tree overfits.
    # Shallow trees = high bias (underfitting)
    # Deep trees   = high variance (overfitting, wiggly boundaries)

    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 18))
    fig2.suptitle("How max_depth Affects Decision Boundaries\n"
                  "(Underfitting → Just Right → Overfitting)",
                  fontsize=16, fontweight='bold', y=0.98)

    for row, (name, X, y) in enumerate(datasets):
        for col, depth in enumerate(depths_to_show):
            tree = DecisionTreeCART(max_depth=depth)
            tree.fit(X, y)
            acc = np.mean(tree.predict(X) == y)

            label = {1: "Underfitting", 3: "Balanced", 10: "Overfitting"}[depth]
            plot_decision_boundary(
                axes2[row, col], tree, X, y,
                f"{name}\ndepth={depth} ({label}) — acc: {acc:.3f}"
            )

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("fig2_depth_effect.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: fig2_depth_effect.png")

    # ═════════════════════════════════════════════════════════════════════
    # FIGURE 3: Train vs Test accuracy across depths (overfitting curve)
    # ═════════════════════════════════════════════════════════════════════
    # This plot drives home the bias-variance tradeoff:
    #   - Train accuracy always increases with depth
    #   - Test accuracy peaks then drops → overfitting!

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("Train vs Test Accuracy Across Tree Depths\n"
                  "(The Overfitting Curve)",
                  fontsize=14, fontweight='bold', y=1.05)

    depth_range = range(1, 21)

    for idx, (name, X, y) in enumerate(datasets):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        train_accs = []
        test_accs = []

        for d in depth_range:
            tree = DecisionTreeCART(max_depth=d)
            tree.fit(X_train, y_train)

            train_accs.append(np.mean(tree.predict(X_train) == y_train))
            test_accs.append(np.mean(tree.predict(X_test) == y_test))

        best_test_depth = depth_range[np.argmax(test_accs)]
        best_test_acc = max(test_accs)

        axes3[idx].plot(depth_range, train_accs, 'o-', color='#1565C0',
                        linewidth=2, label='Train', markersize=4)
        axes3[idx].plot(depth_range, test_accs, 's-', color='#D84315',
                        linewidth=2, label='Test', markersize=4)
        axes3[idx].axvline(x=best_test_depth, color='gray', linestyle='--',
                           alpha=0.5, label=f'Best depth={best_test_depth}')
        axes3[idx].set_title(f"{name}\nBest test acc: {best_test_acc:.3f} "
                             f"@ depth={best_test_depth}",
                             fontsize=11, fontweight='bold')
        axes3[idx].set_xlabel("max_depth", fontsize=10)
        axes3[idx].set_ylabel("Accuracy", fontsize=10)
        axes3[idx].legend(fontsize=9)
        axes3[idx].set_ylim(0.5, 1.05)
        axes3[idx].grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig("fig3_overfitting_curve.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: fig3_overfitting_curve.png")

    # ═════════════════════════════════════════════════════════════════════
    # FIGURE 4: Step-by-step tree growth (animated feel)
    # ═════════════════════════════════════════════════════════════════════
    # Shows depth 1, 2, 3, 4, 5 for the Moons dataset to see how
    # the tree progressively carves up the space.

    fig4, axes4 = plt.subplots(1, 5, figsize=(25, 5))
    fig4.suptitle("How the Tree Grows: Moons Dataset (depth 1→5)\n"
                  "Each depth adds more axis-aligned cuts",
                  fontsize=14, fontweight='bold', y=1.08)

    for i, d in enumerate(range(1, 6)):
        tree = DecisionTreeCART(max_depth=d)
        tree.fit(X_moons, y_moons)
        acc = np.mean(tree.predict(X_moons) == y_moons)
        plot_decision_boundary(
            axes4[i], tree, X_moons, y_moons,
            f"depth={d}  acc={acc:.3f}"
        )

    fig4.tight_layout()
    fig4.savefig("fig4_tree_growth.png", dpi=150, bbox_inches='tight')
    print("✅ Saved: fig4_tree_growth.png")

    plt.show()
    print("\n🎯 What to observe:")
    print("   • Decision boundaries are always AXIS-ALIGNED (vertical/horizontal)")
    print("     → This is a fundamental limitation of single-feature splits!")
    print("   • Deeper trees create more complex (wiggly) boundaries")
    print("   • Our tree and sklearn produce nearly identical boundaries")
    print("   • The overfitting curve shows exactly why max_depth matters\n")


if __name__ == "__main__":
    run_visualization()
