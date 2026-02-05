import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# Load labeled dataset
# ===============================
df = pd.read_csv("features/symbolic_pitch_features_labeled.csv")

# Features and labels
X = df.drop(columns=["emotion", "file"])
y = df["emotion"]

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Train Random Forest model
# ===============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Evaluation
# ===============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# Confusion Matrix Plot
# ===============================
cm = confusion_matrix(y_test, y_pred)
classes = model.classes_

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
plt.tight_layout()
plt.savefig("features/confusion_matrix.png")
print("Confusion matrix saved as features/confusion_matrix.png")

