import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

# Load the feature files
audio_features_csv = "feature_csv/merged_manual_transformer_savee_features.csv"
text_features_csv = "feature_csv/SAVEE_feature_vectors_with_labels.csv"


audio_features = pd.read_csv(audio_features_csv)
text_features = pd.read_csv(text_features_csv)

if len(audio_features) != len(text_features):
    raise ValueError("Mismatch in the number of rows between audio and text features.")


print(f"Audio features shape: {audio_features.shape}")
print(f"Text features shape: {text_features.shape}")

"""1. only using audio  wave to vec features as input for svm - accuracy using baseline model 66% and Best Parameters from GridSearchCV:
{'svm__C': 0.1, 'svm__gamma': 'scale', 'svm__kernel': 'linear'} - accuracy - 69%
"""

audio_features_csv = "feature_csv/saveetransformer_features_with_filenames.csv"
audio_features = pd.read_csv(audio_features_csv)

if "File_Name" not in audio_features.columns or "Emotion" not in audio_features.columns:
    raise ValueError("The dataset must have 'File_Name' and 'Emotion' columns.")

# Extract features and labels
X = audio_features.filter(regex="^Feature_.*").values
y = audio_features["Emotion"].values

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])

svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print(" Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Performing hyperparameter optimization using GridSearchCV
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"],
}

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)


print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\nClassification Report (Best Model):")
print(classification_report(y_test, y_pred_best))

print("Confusion Matrix (Best Model):")
print(confusion_matrix(y_test, y_pred_best))

"""2. wave to vec audio feature, manual audio features and text features in svm  - accuracy 58%"""

audio_features_csv = "feature_csv/merged_manual_transformer_savee_features.csv"
text_features_csv = "feature_csv/SAVEE_feature_vectors_with_labels.csv"


audio_features = pd.read_csv(audio_features_csv)
text_features = pd.read_csv(text_features_csv)


if "File_Name" not in audio_features.columns or "File_Name" not in text_features.columns:
    raise ValueError("Both datasets must have a 'File_Name' column for merging.")
if "Emotion_transformer" not in audio_features.columns or "Emotion" not in text_features.columns:
    raise ValueError("Both datasets must have the required emotion columns for alignment.")


merged_features = pd.merge(audio_features, text_features, on="File_Name", suffixes=("_audio", "_text"))

matched_features = merged_features[merged_features["Emotion_transformer"] == merged_features["Emotion"]]

file_names = matched_features["File_Name"]
emotions = matched_features["Emotion"]


aligned_audio_features = matched_features.filter(regex="^Feature_.*_(manual|transformer)$")
aligned_text_features = matched_features.filter(regex="^Feature_\\d+$")


combined_features_df = pd.concat(
    [file_names, emotions, aligned_audio_features, aligned_text_features], axis=1
)


print("Combined DataFrame preview:")
print(combined_features_df.head())
print(f"Combined DataFrame shape: {combined_features_df.shape}")


X = combined_features_df.drop(columns=["File_Name", "Emotion"]).values
y = combined_features_df["Emotion"].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])


svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

"""3. only manual audio features and text features in svm - accuracy 50%"""

audio_features_csv = "feature_csv/manual_features_with_filenames_final.csv"
text_features_csv = "feature_csv/SAVEE_feature_vectors_with_labels.csv"


audio_features = pd.read_csv(audio_features_csv)
text_features = pd.read_csv(text_features_csv)


if "File_Name" not in audio_features.columns or "File_Name" not in text_features.columns:
    raise ValueError("Both datasets must have a 'File_Name' column for merging.")
if "Emotion" not in audio_features.columns or "Emotion" not in text_features.columns:
    raise ValueError("Both datasets must have an 'Emotion' column for alignment.")


merged_features = pd.merge(audio_features, text_features, on="File_Name")


matched_features = merged_features[merged_features["Emotion_x"] == merged_features["Emotion_y"]]


file_names = matched_features["File_Name"]
emotions = matched_features["Emotion_x"]


aligned_audio_features = matched_features.filter(regex="^Feature_\\d+_audio$")
aligned_text_features = matched_features.filter(regex="^Feature_\\d+$")


combined_features_df = pd.concat(
    [file_names, emotions, aligned_audio_features, aligned_text_features], axis=1
)


print("Combined DataFrame :")
print(combined_features_df.head())
print("Columns in Combined DataFrame:")
print(combined_features_df.columns)
print(f"Combined DataFrame shape: {combined_features_df.shape}")


X = combined_features_df.drop(columns=["File_Name", "Emotion_x"]).values
y = combined_features_df["Emotion_x"].values


imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])


svm_pipeline.fit(X_train, y_train)


y_pred = svm_pipeline.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

"""4. wave to vec audio features and text features as input in svm - 64% using svm and using grid search - Best Parameters: {'svm__C': 0.1, 'svm__gamma': 'scale', 'svm__kernel': 'linear'} - 67%"""

audio_features_csv = "feature_csv/saveetransformer_features_with_filenames.csv"
text_features_csv = "feature_csv/SAVEE_feature_vectors_with_labels.csv"

audio_features = pd.read_csv(audio_features_csv)
text_features = pd.read_csv(text_features_csv)

if "File_Name" not in audio_features.columns or "File_Name" not in text_features.columns:
    raise ValueError("Both datasets must have a 'File_Name' column for merging.")
if "Emotion" not in audio_features.columns or "Emotion" not in text_features.columns:
    raise ValueError("Both datasets must have an 'Emotion' column for alignment.")

audio_features.rename(columns=lambda x: f"{x}_audio" if x.startswith("Feature_") else x, inplace=True)
text_features.rename(columns=lambda x: f"{x}_text" if x.startswith("Feature_") else x, inplace=True)

merged_features = pd.merge(audio_features, text_features, on="File_Name")

matched_features = merged_features[merged_features["Emotion_x"] == merged_features["Emotion_y"]]

file_names = matched_features["File_Name"]
emotions = matched_features["Emotion_x"]

aligned_audio_features = matched_features.filter(regex="^Feature_.*_audio$")
aligned_text_features = matched_features.filter(regex="^Feature_.*_text$")


combined_features_df = pd.concat(
    [file_names, emotions, aligned_audio_features, aligned_text_features], axis=1
)

print("Combined DataFrame :")
print(combined_features_df.head())
print("Columns in Combined DataFrame:")
print(combined_features_df.columns)
print(f"Combined DataFrame shape: {combined_features_df.shape}")


X = combined_features_df.drop(columns=["File_Name", "Emotion_x"]).values
y = combined_features_df["Emotion_x"].values


imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])


svm_pipeline.fit(X_train, y_train)


y_pred = svm_pipeline.predict(X_test)


print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)


best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)


print("Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best))

print("Confusion Matrix (Best Model):")
print(confusion_matrix(y_test, y_pred_best))

"""Now, used various svm hyper tuning methods to increase accuracy. Here, the classification is done taking input as wave to vec and word to vec features.

1. Using GridSearchCV for Hyperparameter Tuning - accuracy 67%
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": [0.1, 1, 10]
}

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="accuracy", verbose=3)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best parameters:", grid_search.best_params_)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

"""2. Feature Selection Using Recursive Feature Elimination - accuracy 31% so not useable"""

from sklearn.feature_selection import RFE

svm = SVC(kernel="linear", random_state=42)
rfe = RFE(estimator=svm, n_features_to_select=10)
X_train_selected = rfe.fit_transform(X_train, y_train)
X_test_selected = rfe.transform(X_test)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])


svm_pipeline.fit(X_train_selected, y_train)


y_pred = svm_pipeline.predict(X_test_selected)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

"""3.Handling Imbalanced Data Using SMOTE - accuracy - 64%"""

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", probability=True, random_state=42))
])

svm_pipeline.fit(X_train_resampled, y_train_resampled)

y_pred = svm_pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))