import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

class LSTMEstimator:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        # No training needed (already trained)
        return self

    def predict(self, X):
        # X is 2D => shape (batch, features)
        # LSTM model needs 3D => shape (batch, 1, features)
        X_3d = X.reshape((X.shape[0], 1, X.shape[1]))
        preds = self.model.predict(X_3d)
        return (preds > 0.5).astype(int).ravel()

    def score(self, X, y):
        """Default scikit-learn scoring method (accuracy)."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # simple accuracy

def main():
    # 1. Load preprocessed data
    data = pd.read_csv("data/processed_data.csv")
    X = data.drop(columns=["Label"]).values
    y = data["Label"].values
    feature_names = data.drop(columns=["Label"]).columns

    # 2. Reshape for LSTM input (samples, timesteps=1, features)
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y, test_size=0.2, random_state=42
    )

    # 4. Load Trained LSTM Model
    model = tf.keras.models.load_model("models/lstm_scada_model.keras")

    # 5. Flatten X_test to 2D for scikit-learn
    X_test_flat = X_test.reshape((X_test.shape[0], X_test.shape[2]))

    # 6. Wrap the LSTM model in our scikit-learn–style estimator
    wrapped_lstm = LSTMEstimator(model=model)

    # 7. Compute permutation importance
    print("Computing Permutation Importance... (may take a while)")
    r = permutation_importance(
        estimator=wrapped_lstm,
        X=X_test_flat,
        y=y_test,
        n_repeats=5,
        random_state=42
    )

    importances = r.importances_mean
    stds = r.importances_std

    # 8. Sort features by importance (descending)
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_stds = stds[indices]
    sorted_features = feature_names[indices]

    # 9. Print feature ranking to console
    print("\n=== Feature Ranking (Permutation Importance) ===")
    for rank, (feature, imp, s) in enumerate(
        zip(sorted_features, sorted_importances, sorted_stds), start=1
    ):
        print(f"{rank}. {feature} => Mean: {imp:.6f} ± {s:.6f}")

    # 10. Generate a heatmap
    importances_matrix = sorted_importances.reshape(1, -1)

    plt.figure(figsize=(12, 2.5))
    sns.heatmap(
        importances_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".6f",
        xticklabels=sorted_features,
        yticklabels=["Importance"]
    )
    plt.title("Permutation Importance Heatmap - LSTM Model")
    plt.xlabel("Features (sorted by importance)")
    plt.ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
