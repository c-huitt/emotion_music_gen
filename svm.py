import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

# File paths
audio_feature_files = [
    "feature_csv/savee_features.csv",
    "feature_csv/tess_features_audio.csv",
    "feature_csv/crema_features_audio.csv"
]

text_feature_files = [
    "feature_csv/SAVEE_feature_vectors_with_labels.csv",
    "feature_csv/tess_features_emotions.csv",
    "feature_csv/crema_features_emotions.csv"
]

# Combine all datasets
combined_features = []
file_names = []
emotions = []

for audio_file, text_file in zip(audio_feature_files, text_feature_files):
    # Load audio and text features
    audio_features = pd.read_csv(audio_file)
    text_features = pd.read_csv(text_file)

    # Ensure alignment of files by filename
    if "File_Name" not in audio_features.columns:
        audio_features["File_Name"] = text_features["File_Name"]

    # Check emotion alignment
    for i in range(len(audio_features)):
        if audio_features.loc[i, "Emotion"] == text_features.loc[i, "Emotion"]:
            combined_row = {
                "File_Name": text_features.loc[i, "File_Name"],
                "Emotion": audio_features.loc[i, "Emotion"],
            }
            # Combine audio and text features
            combined_row.update(audio_features.iloc[i, :].filter(like="Feature_").to_dict())
            combined_row.update(text_features.iloc[i, :].filter(like="Feature_").to_dict())
            combined_features.append(combined_row)

# Create a DataFrame for the combined features
combined_df = pd.DataFrame(combined_features)
file_names = combined_df["File_Name"].tolist()
emotions = combined_df["Emotion"].tolist()

# Extract feature columns
feature_columns = [col for col in combined_df.columns if col.startswith("Feature_")]
X = combined_df[feature_columns].values
y = emotions

print(f"Combined feature matrix shape: {X.shape}")
print(f"Labels shape: {len(y)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test, file_names_train, file_names_test = train_test_split(
    X, y, file_names, test_size=0.2, random_state=42, stratify=y
)

# Create a pipeline with scaling and SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Scale features for better SVM performance
    ("svm", SVC(kernel="linear", probability=True, random_state=42))  # Linear kernel SVM
])

# Train the SVM model
svm_pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_pipeline.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create output DataFrame with filenames and predicted labels
output_df = pd.DataFrame({
    "File_Name": file_names_test,
    "True_Emotion": y_test,
    "Predicted_Emotion": y_pred
})

# Save output to CSV
output_csv_path = "predicted_emotions/predicted_emotions.csv"
output_df.to_csv(output_csv_path, index=False)
print(f"Predicted emotions saved to {output_csv_path}")

cm = confusion_matrix(y_test, y_pred)  # Confusion matrix
class_labels = sorted(set(y_test))

# Calculate accuracy for each emotion
accuracies = {}
for i, label in enumerate(class_labels):
    true_positive = cm[i, i]
    total_samples = cm[i, :].sum()
    accuracy = true_positive / total_samples if total_samples > 0 else 0
    accuracies[label] = accuracy


emotions = list(accuracies.keys())
accuracy_values = list(accuracies.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(emotions, accuracy_values, color='skyblue', edgecolor='black')


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

# Adding titles and labels
plt.title("Accuracy for Each Emotion", fontsize=14)
plt.xlabel("Emotion", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/svm_accuracy_each_emotion.png")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y), yticklabels=set(y))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.savefig("plots/svm_confusion_matrix.png")

report = classification_report(y_test, y_pred, output_dict=True)

# Extract metrics for each emotion
emotions = list(report.keys())[:-3]
metrics = {metric: [report[emotion][metric] for emotion in emotions] for metric in ["precision", "recall", "f1-score"]}

# Creating grouped bar chart
x = np.arange(len(emotions))
width = 0.25

plt.figure(figsize=(12, 6))

# Plotting precision, recall, and F1-score for each emotion
for i, (metric_name, values) in enumerate(metrics.items()):
    plt.bar(x + (i - 1) * width, values, width, label=metric_name.capitalize(), edgecolor="black")

# Adding labels and legend
plt.xticks(x, emotions, rotation=45)
plt.xlabel("Emotion", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Performance Metrics by Emotion", fontsize=14)
plt.legend(loc="upper left", fontsize=10)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("plots/svm_performance_matrix.png")

"""SVM - SAVEE CREMA TESS USING GRID SEARCH"""

warnings.filterwarnings('ignore')

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Encoded Labels:", np.unique(y_encoded))
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


X_train, X_test, y_train, y_test, file_names_train, file_names_test = train_test_split(
    X, y_encoded, file_names, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")

# Handling class imbalance with SMOTE
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE, training set shape: {X_train_resampled.shape}")

# Creating a pipeline with scaling, PCA, and SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("svm", SVC(probability=True, random_state=42))
])

# Define Hyperparameter Distribution for RandomizedSearchCV
param_dist = {
    "svm__C": uniform(loc=0.1, scale=100),
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__degree": randint(2, 6),
    "svm__gamma": ["scale", "auto"]
}

# Initializing RandomizedSearchCV
random_search_svm = RandomizedSearchCV(
    estimator=svm_pipeline,
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',
    random_state=42
)

# Train the SVM model with RandomizedSearchCV
print("\nStarting RandomizedSearchCV for SVM with PCA and SMOTE...")
random_search_svm.fit(X_train_resampled, y_train_resampled)
print("RandomizedSearchCV completed.")

# Best Parameters and Cross-validation Score
print("\nBest Parameters for SVM:", random_search_svm.best_params_)
print("Best Cross-validation Accuracy for SVM:", random_search_svm.best_score_)


y_pred = random_search_svm.predict(X_test)

# Evaluating the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Creating output DataFrame with filenames and predicted labels
output_df = pd.DataFrame({
    "File_Name": file_names_test,
    "True_Emotion": label_encoder.inverse_transform(y_test),
    "Predicted_Emotion": label_encoder.inverse_transform(y_pred)
})