#Hyperparameter tuning of models in Appendix A
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
seed = 42
np.random.seed(seed)

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=20,
    n_redundant=0,
    n_clusters_per_class=2,
    n_classes=2,
    flip_y=0.15,
    weights=[0.9, 0.1],
    random_state=seed
)

# Add random noise variables
X_random = np.random.randn(1000, 20)
X_extended = np.hstack((X, X_random))

df = pd.DataFrame(
    X_extended,
    columns=[f"feature_{i+1}" for i in range(X_extended.shape[1])]
)
df["y"] = y
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["y"]),
    df["y"],
    test_size=0.1,
    random_state=seed
)
models = {

    "Random Forest": (
        RandomForestClassifier(random_state=seed),
        {
            "n_estimators": [100, 200],
            "max_features": ["sqrt", "log2"]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=seed),
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    ),

    "Extra Trees": (
        ExtraTreesClassifier(random_state=seed),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, None]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=seed),
        {
            "max_depth": [3, 5, None],
            "criterion": ["gini", "entropy"]
        }
    ),

    "SVM": (
        SVC(probability=True, random_state=seed),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    ),

    "Logistic Regression": (
        LogisticRegression(random_state=seed),
        {
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    ),

    "MLP": (
        MLPClassifier(max_iter=500, random_state=seed),
        {
            "hidden_layer_sizes": [(128,), (128,)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001]
        }
    ),

    "Gaussian NB": (
        GaussianNB(),
        {
            "var_smoothing": [1e-9, 1e-8, 1e-7]
        }
    ),

    "LDA": (
        LinearDiscriminantAnalysis(),
        {
            "solver": ["svd", "lsqr"]
        }
    ),

    "XGBoost": (
        xgb.XGBClassifier(
            random_state=seed,
            eval_metric="logloss"
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0]
        }
    )
}

best_models = {}
test_auc_scores = {}

auc_scorer = make_scorer(
    roc_auc_score,
    response_method="predict_proba"
)

for model_name, (model, param_grid) in models.items():

    print("=" * 60)
    print(f"Running GridSearchCV for {model_name}...")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=auc_scorer,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    best_models[model_name] = best_model

    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    test_auc_scores[model_name] = test_auc

    print("Best Parameters:")
    print(grid_search.best_params_)

    print(f"Best CV AUC : {grid_search.best_score_:.4f}")
    print(f"Test AUC    : {test_auc:.4f}")
    print()

# ======================================================
# Final Results
# ======================================================

print("\n================ Final Test AUC Scores ================\n")

for model_name, auc in sorted(test_auc_scores.items(),
                              key=lambda x: x[1],
                              reverse=True):
    print(f"{model_name:25s}: {auc:.4f}")
