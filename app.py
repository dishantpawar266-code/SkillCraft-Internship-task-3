import os
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog


# =========================
# SETTINGS
# =========================

TRAIN_DIR = "train"
TEST_DIR = "test1"

IMAGE_SIZE = (64, 64)


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# =========================
# LOAD TRAINING DATA
# =========================

X_train = []
y_train = []

print("Loading training images...")

for filename in os.listdir(TRAIN_DIR):

    filepath = os.path.join(TRAIN_DIR, filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    features = extract_features(filepath)

    if features is None:
        continue

    X_train.append(features)

    # Label from filename
    if filename.lower().startswith("cat"):
        y_train.append(0)

    elif filename.lower().startswith("dog"):
        y_train.append(1)


X_train = np.array(X_train)
y_train = np.array(y_train)

print("Training images:", len(X_train))


# =========================
# TRAIN SVM
# =========================

print("\nTraining SVM...")

model = SVC(kernel="rbf")

model.fit(X_train, y_train)

print("SVM training completed!")


# =========================
# PREDICT TEST IMAGES
# =========================

print("\n==============================")
print("TEST IMAGE PREDICTIONS")
print("==============================")

correct = 0
total = 0

for filename in os.listdir(TEST_DIR):

    filepath = os.path.join(TEST_DIR, filename)

    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    features = extract_features(filepath)

    if features is None:
        continue

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 0:
        predicted_class = "CAT"
    else:
        predicted_class = "DOG"

    print(f"{filename}  -->  {predicted_class}")

    # If test filename contains actual label,
    # calculate accuracy
    if filename.lower().startswith("cat"):
        actual = 0
        total += 1

        if prediction == actual:
            correct += 1

    elif filename.lower().startswith("dog"):
        actual = 1
        total += 1

        if prediction == actual:
            correct += 1


# =========================
# ACCURACY
# =========================

if total > 0:

    accuracy = (correct / total) * 100

    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Correct Predictions : {correct}")
    print(f"Total Test Images  : {total}")
    print(f"Accuracy            : {accuracy:.2f}%")