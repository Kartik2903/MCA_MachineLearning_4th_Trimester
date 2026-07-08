import json
import os

cells = [
    # Metadata / Intro
    {"cell_type": "markdown", "metadata": {}, "source": ["# Lab 4: K-Nearest Neighbours (KNN) Classification\n", "### Breast Cancer Dataset and Comparison with Regression Evaluation Metrics"]},
    
    # Task 1
    {"cell_type": "markdown", "metadata": {}, "source": ["## Task 1: Data Preparation\n", "Load dataset, check structure, and apply feature scaling."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.datasets import load_breast_cancer\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "# Load dataset\n",
        "data = load_breast_cancer()\n",
        "df = pd.DataFrame(data.data, columns=data.feature_names)\n",
        "df['target'] = data.target # 0: Malignant, 1: Benign\n",
        "\n",
        "# Check structure, missing values, duplicates\n",
        "print('Shape:', df.shape)\n",
        "print('Missing Values:', df.isnull().sum().sum())\n",
        "print('Duplicates:', df.duplicated().sum())\n",
        "\n",
        "# Feature scaling\n",
        "X = df.drop(columns=['target'])\n",
        "y = df['target']\n",
        "scaler = StandardScaler()\n",
        "X_scaled = scaler.fit_transform(X)\n",
        "\n",
        "print('Data Scaled using StandardScaler.')"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["**Importance of Feature Scaling:** KNN is a distance-based algorithm. If features are on different scales (e.g., one ranges 0-1 and another 0-1000), the feature with the larger range will dominate the distance computation. StandardScaler ensures all features contribute equally."]},
    
    # Task 2
    {"cell_type": "markdown", "metadata": {}, "source": ["## Task 2: Train-Test Split Analysis\n", "Compare performance across 80:20, 70:30, and 90:10 splits."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "splits = [0.2, 0.3, 0.1]\n",
        "split_names = ['80:20', '70:30', '90:10']\n",
        "\n",
        "for test_size, name in zip(splits, split_names):\n",
        "    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=test_size, random_state=42)\n",
        "    knn = KNeighborsClassifier(n_neighbors=5)\n",
        "    knn.fit(X_tr, y_tr)\n",
        "    acc = accuracy_score(y_te, knn.predict(X_te))\n",
        "    print(f'Split {name} - Accuracy: {acc:.4f}')\n"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["**Analysis:** Changing the train-test split affects model stability. A smaller training set might underfit, while a smaller test set leads to higher variance in evaluation metrics. An 80:20 split is generally a solid balance for stability and generalization."]},
    
    # Task 3
    {"cell_type": "markdown", "metadata": {}, "source": ["## Task 3: KNN Model with Heuristic K Selection\n", "Compute initial K using heuristic rule (K = √n). Plot accuracy for nearby K values."]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "import math\n",
        "\n",
        "# 80:20 split for remaining tasks\n",
        "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
        "\n",
        "n_train = X_train.shape[0]\n",
        "heuristic_k = math.floor(math.sqrt(n_train))\n",
        "if heuristic_k % 2 == 0: heuristic_k -= 1 # Ensure odd K for binary classification\n",
        "print(f'Heuristic K (sqrt(n)): {heuristic_k}')\n",
        "\n",
        "k_values = range(max(1, heuristic_k - 10), heuristic_k + 11, 2)\n",
        "accuracies = []\n",
        "\n",
        "for k in k_values:\n",
        "    knn = KNeighborsClassifier(n_neighbors=k)\n",
        "    knn.fit(X_train, y_train)\n",
        "    accuracies.append(accuracy_score(y_test, knn.predict(X_test)))\n",
        "\n",
        "plt.figure(figsize=(8, 5))\n",
        "plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b')\n",
        "plt.title('Accuracy vs. K Value')\n",
        "plt.xlabel('K Value')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.grid()\n",
        "plt.show()"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "### Distance Metrics\n",
        "- **Euclidean Distance:** Shortest straight-line distance between two points. Suitable for continuous numerical data without high dimensions.\n",
        "- **Manhattan Distance:** Sum of absolute differences across all dimensions. Suitable for high-dimensional data or grid-like paths."
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["### Decision Boundary Visualization (using top 2 PCA components)"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from sklearn.decomposition import PCA\n",
        "from matplotlib.colors import ListedColormap\n",
        "\n",
        "# Reduce to 2 dimensions for visualization\n",
        "pca = PCA(n_components=2)\n",
        "X_pca = pca.fit_transform(X_scaled)\n",
        "X_tr_pca, X_te_pca, y_tr_pca, y_te_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)\n",
        "\n",
        "def plot_decision_boundary(k):\n",
        "    knn_pca = KNeighborsClassifier(n_neighbors=k)\n",
        "    knn_pca.fit(X_tr_pca, y_tr_pca)\n",
        "    \n",
        "    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1\n",
        "    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1\n",
        "    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))\n",
        "    \n",
        "    Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])\n",
        "    Z = Z.reshape(xx.shape)\n",
        "    \n",
        "    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FF9999', '#99FF99']))\n",
        "    plt.scatter(X_te_pca[:, 0], X_te_pca[:, 1], c=y_te_pca, edgecolor='k', cmap=ListedColormap(['red', 'green']))\n",
        "    plt.title(f'KNN Decision Boundary (K={k})')\n",
        "\n",
        "plt.figure(figsize=(15, 10))\n",
        "for i, k in enumerate([1, 5, 10, 20]):\n",
        "    plt.subplot(2, 2, i+1)\n",
        "    plot_decision_boundary(k)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": ["**Analysis:** Small K (K=1) leads to highly complex decision boundaries (overfitting and sensitive to noise). As K increases, the boundary becomes smoother and more generalized."]},
    
    # Task 4
    {"cell_type": "markdown", "metadata": {}, "source": ["## Task 4: Cross Validation"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "\n",
        "cv_scores = []\n",
        "for k in k_values:\n",
        "    knn = KNeighborsClassifier(n_neighbors=k)\n",
        "    scores = cross_val_score(knn, X_scaled, y, cv=5) # 5-Fold CV\n",
        "    cv_scores.append(scores.mean())\n",
        "\n",
        "optimal_k = k_values[np.argmax(cv_scores)]\n",
        "print(f'Optimal K from Cross-Validation: {optimal_k}')\n",
        "\n",
        "plt.figure(figsize=(8, 5))\n",
        "plt.plot(k_values, cv_scores, marker='s', color='orange')\n",
        "plt.title('5-Fold Cross Validation Accuracy vs K')\n",
        "plt.xlabel('K Value')\n",
        "plt.ylabel('Mean CV Accuracy')\n",
        "plt.grid()\n",
        "plt.show()"
    ]},
    
    # Task 5
    {"cell_type": "markdown", "metadata": {}, "source": ["## Task 5: Classification Evaluation"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc\n",
        "\n",
        "final_knn = KNeighborsClassifier(n_neighbors=optimal_k)\n",
        "final_knn.fit(X_train, y_train)\n",
        "y_pred = final_knn.predict(X_test)\n",
        "y_prob = final_knn.predict_proba(X_test)[:, 1]\n",
        "\n",
        "print('Classification Report:\\n', classification_report(y_test, y_pred))\n",
        "\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)\n",
        "plt.title('Confusion Matrix')\n",
        "plt.show()\n",
        "\n",
        "fpr, tpr, thresholds = roc_curve(y_test, y_prob)\n",
        "roc_auc = auc(fpr, tpr)\n",
        "\n",
        "plt.figure(figsize=(6, 5))\n",
        "plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')\n",
        "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
        "plt.title('Receiver Operating Characteristic (ROC)')\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.legend(loc='lower right')\n",
        "plt.show()"
    ]},
    
    # Task 6
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Task 6: Comparative Study with Regression (Lab 3 Integration)\n",
        "\n",
        "### Differences Between Evaluation Metrics:\n",
        "- **Error-based evaluation (Regression):** Measures continuous distances/differences between actual and predicted values (e.g., RMSE, MAE). The goal is to minimize magnitude of error.\n",
        "- **Decision-based evaluation (Classification):** Measures discrete correctness (right vs. wrong class). Metrics like Precision and Recall care about categorical outcomes.\n",
        "\n",
        "### Metric Comparisons:\n",
        "- **R² Score vs Accuracy:** R² measures how much variance is explained by a regression model. Accuracy measures the percentage of correctly predicted classes.\n",
        "- **RMSE vs F1 Score:** RMSE penalizes large numerical errors in prediction. F1 Score balances Precision and Recall, crucial for imbalanced classification.\n",
        "- **MAE vs Confusion Matrix:** MAE gives average absolute error distance. A Confusion Matrix gives a detailed breakdown of exact misclassifications (False Positives vs False Negatives).\n",
        "\n",
        "### Inference Requirement (Healthcare Analytics):\n",
        "- **Regression Metrics:** Measure the magnitude of continuous predictions (e.g., tumor size prediction).\n",
        "- **Classification Metrics:** Measure decision correctness (e.g., Benign vs Malignant).\n",
        "- **Why is accuracy insufficient?** In medical diagnosis, data is often imbalanced. A model predicting everyone as \"Benign\" might have 90% accuracy but miss all cancer cases.\n",
        "- **Why are recall and ROC-AUC more relevant?** Recall minimizes False Negatives (missing a cancer diagnosis is deadly). ROC-AUC evaluates the model's ability to distinguish between classes across different thresholds."
    ]},
    
    # Task 7
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Task 7: Analytical Questions\n",
        "\n",
        "1. **Why is KNN called a lazy learning algorithm?** It does not build an explicit model during the training phase. It simply stores the training dataset and performs all distance computations during the prediction phase.\n",
        "2. **Why is feature scaling required in KNN?** KNN uses distance metrics (like Euclidean). Unscaled features with larger ranges will disproportionately influence the distance calculation.\n",
        "3. **Explain heuristic K selection using √n rule:** Setting K equal to the square root of the number of training samples (and making it odd) provides a statistically sound baseline that balances complexity and smoothing.\n",
        "4. **Why is cross-validation more reliable than a single train-test split?** It trains and evaluates the model on multiple different subsets, ensuring the performance metrics are not just dependent on one \"lucky\" or \"unlucky\" random split.\n",
        "5. **How does K affect bias-variance trade-off?** Small K = Low Bias, High Variance (Overfitting). Large K = High Bias, Low Variance (Underfitting).\n",
        "6. **Why is recall more important than accuracy in cancer prediction?** Recall focuses on capturing all actual positive (Malignant) cases. A False Negative (telling a sick patient they are healthy) is much worse than a False Positive.\n",
        "7. **What is the limitation of very large K values?** If K is too large, it considers points that are too far away, and the prediction will just default to the majority class in the entire dataset (underfitting)."
    ]},
    
    # Conclusion
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Conclusion\n",
        "- **Optimal K:** Selected through both the heuristic baseline ($\\sqrt{n}$) and validated through K-Fold Cross Validation.\n",
        "- **Train-Test Splits:** Showcased that smaller training sets cause variability, and 80:20 provides a stable evaluation.\n",
        "- **Performance:** The model performs highly effectively on the Breast Cancer dataset, with a focus on Recall and ROC-AUC for medical reliability.\n",
        "- **Regression vs Classification:** Lab 3 metrics aimed to minimize continuous error, while Lab 4 metrics optimize class decision boundaries and correct categorization."
    ]}
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('/home/kartik/Documents/Machine_learning/Lab_4/Lab4.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Lab4.ipynb generated successfully.")
