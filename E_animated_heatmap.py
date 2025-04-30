import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

class LSTMEstimator:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        X_3d = X.reshape((X.shape[0], 1, X.shape[1]))
        preds = self.model.predict(X_3d)
        return (preds > 0.5).astype(int).ravel()

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  # accuracy

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

    # 7. Prepare folder for animation frames
    os.makedirs("heatmap_frames", exist_ok=True)
    num_slices = 10  # Number of time steps for animation
    frames = []

    # 8. Generate heatmaps over time
    for i in range(num_slices):
        print(f"Computing Permutation Importance (Step {i+1}/{num_slices})...")
        subset = X_test_flat[i * 500 : (i + 1) * 500]  # Slice data over time
        subset_labels = y_test[i * 500 : (i + 1) * 500]
        
        r = permutation_importance(
            estimator=wrapped_lstm,
            X=subset,
            y=subset_labels,
            n_repeats=5,
            random_state=42
        )
        
        importances = r.importances_mean
        indices = np.argsort(importances)[::-1]
        sorted_importances = importances[indices]
        sorted_features = feature_names[indices]
        
        # 9. Plot heatmap frame
        plt.figure(figsize=(12, 2.5))
        sns.heatmap(
            sorted_importances.reshape(1, -1),
            cmap="coolwarm",
            annot=True,
            fmt=".6f",
            xticklabels=sorted_features,
            yticklabels=[f"Time Step {i+1}"]
        )
        plt.title("Permutation Importance Heatmap - LSTM Model (Temporal Analysis)")
        plt.xlabel("Features (sorted by importance)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        frame_path = f"heatmap_frames/frame_{i}.png"
        plt.savefig(frame_path)
        plt.close()
        frames.append(imageio.imread(frame_path))
    
    # 10. Save animation
    gif_path = "animated_heatmap.gif"
    imageio.mimsave(gif_path, frames, duration=1)
    print(f" Animation saved as {gif_path}")

if __name__ == "__main__":
    main()