import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

# Load the processed dataset
data = pd.read_csv("data/processed_data.csv")
X = data.drop(columns=["Label"]).values
Y = data["Label"].values

# Reshape X for LSTM input
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Load trained model
model = tf.keras.models.load_model("models/lstm_scada_model.keras")

# Make predictions
Y_pred = (model.predict(X_test) > 0.5).astype(int)

# Compute confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - LSTM Model")
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(Y_test, model.predict(X_test))
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSTM Model")
plt.legend()
plt.show()
