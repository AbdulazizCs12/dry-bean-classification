import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# ==========================
# Load Dataset
# ==========================

data = pd.read_excel("Dry_Bean_Dataset.xlsx")

print("Dataset Shape:", data.shape)
print("Missing Values:", data.isnull().sum().sum())


# ==========================
# Features and Target
# ==========================

X = data.drop("Class", axis=1)
y = data["Class"]


# ==========================
# Encode Target Classes
# ==========================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClasses:")

for number, class_name in enumerate(label_encoder.classes_):
    print(number, "=", class_name)


# ==========================
# Train / Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ==========================
# Neural Network
# ==========================

model = Sequential([
    Input(shape=(16,)),

    Dense(32, activation="relu"),
    Dense(16, activation="relu"),

    Dense(7, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nNeural Network Architecture:")
model.summary()


# ==========================
# Train Model
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_split=0.2
)


# ==========================
# Evaluate Model
# ==========================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================
# Predictions
# ==========================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)


# ==========================
# Classification Results
# ==========================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ==========================
# Accuracy Plot
# ==========================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("accuracy_curve.png")
plt.close()


# ==========================
# Loss Plot
# ==========================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig("loss_curve.png")
plt.close()

print("\nTraining plots saved successfully.")