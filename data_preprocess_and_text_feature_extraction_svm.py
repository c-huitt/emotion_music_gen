import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Load the Whisper model
import whisper
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec

"""**DATASET PREPROCESSING**

**RAVDESS**
"""

Ravdess = "datasets/Ravdess"

base_dir = "datasets/Ravdess"

audio_extensions = ["*.wav"]

# Initialize an empty list to store all audio files
audio_files = []

# Loop through each Actor folder from Actor_01 to Actor_24
for i in range(1, 25):  # Loop from 1 to 24
    actor_folder = os.path.join(base_dir, f"Actor_{i:02d}")  # Format folder names like Actor_01, Actor_02...

    # Loop through each extension and search for files in the current actor folder
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(actor_folder, ext)))  # Collect .wav files

# Print the total number of audio files across all folders
print(f"Total number of audio files: {len(audio_files)}")

for filename in os.listdir(Ravdess):
    print(filename)

ravdess_directory_list = os.listdir(Ravdess)  # List all directories in Ravdess (each actor)
emotion_df = []

for dir in ravdess_directory_list:
    actor = os.listdir(os.path.join(Ravdess, dir))  # List all files for each actor
    for wav in actor:
        # Check if the file is a .wav file and follows the expected naming pattern
        if wav.endswith(".wav"):
            info = wav.partition(".wav")[0].split("-")
            if len(info) > 2:  # Ensure there are at least 3 parts
                emotion = int(info[2])  # Extract emotion code as an integer
                # Append the emotion and full path of the .wav file to emotion_df
                emotion_df.append((emotion, os.path.join(Ravdess, dir, wav)))

# Print the collected data for inspection
print(emotion_df)

Ravdess_df = pd.DataFrame.from_dict(emotion_df)
Ravdess_df.rename(columns={1 : "Path", 0 : "Emotion"}, inplace=True)

Ravdess_df.Emotion.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()

"""**CREMA**"""

Crema = "datasets/Crema"

# Set the directory path where audio files are stored
audio_dir = "datasets/Crema"

# Define the audio file extensions to look for
audio_extensions = ["*.wav"]  # Only looking for .wav files

# Count audio files with the specified extensions
audio_files = []
for ext in audio_extensions:
    audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))  # Correct loop over extensions

# Print the total number of audio files
print(f"Total number of audio files: {len(audio_files)}")

for filename in os.listdir(Crema):
    print(filename)

emotion_df = []

for wav in os.listdir(Crema):
    info = wav.partition(".wav")[0].split("_")
    if info[2] == 'SAD':
        emotion_df.append(("sad", Crema + "/" + wav))
    elif info[2] == 'ANG':
        emotion_df.append(("angry", Crema + "/" + wav))
    elif info[2] == 'DIS':
        emotion_df.append(("disgust", Crema + "/" + wav))
    elif info[2] == 'FEA':
        emotion_df.append(("fear", Crema + "/" + wav))
    elif info[2] == 'HAP':
        emotion_df.append(("happy", Crema + "/" + wav))
    elif info[2] == 'NEU':
        emotion_df.append(("neutral", Crema + "/" + wav))
    else:
        emotion_df.append(("surprise", Crema + "/" + wav))


Crema_df = pd.DataFrame.from_dict(emotion_df)
Crema_df.rename(columns={1 : "Path", 0 : "Emotion"}, inplace=True)
Crema_df.head()

"""**TESS**"""

Tess = "datasets/Tess"

# Set the directory path where audio files are stored
audio_dir = "datasets/Tess"

# Define the audio file extensions to look for
audio_extensions = ["*.wav"]  # Only looking for .wav files

# Count audio files with the .wav extension
audio_files = []
for ext in audio_extensions:
    audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))  # Correct loop over extensions

# Print the total number of audio files
print(f"Total number of audio files: {len(audio_files)}")

for filename in os.listdir(Tess):
    print(filename)

# Initialize an empty list to store emotion and path data
emotion_df = []

for wav in os.listdir(Tess):
    if wav.endswith(".wav"):  # Process only .wav files
        info = wav.partition(".wav")[0].split("_")
        emo = info[2]
        # Map 'ps' to 'surprise' for consistency
        emotion_label = "surprise" if emo == "ps" else emo
        emotion_df.append((emotion_label, os.path.join(Tess, wav)))

# Convert the list to a DataFrame
Tess_df = pd.DataFrame.from_dict(emotion_df)

Tess_df.rename(columns={0: "Emotion", 1: "Path"}, inplace=True)

Tess_df.head()

"""**SAVEE**"""

# Define the path for the SAVEE dataset
Savee = "datasets/Savee"

# Get the list of files in the directory
savee_directory_list = os.listdir(Savee)

# Count the number of .wav files in the SAVEE dataset
num_wav_files = sum(1 for wav in savee_directory_list if wav.endswith(".wav"))
print(f"Total number of .wav files in SAVEE dataset: {num_wav_files}")

savee_directiory_list = os.listdir(Savee)

emotion_df = []

for wav in savee_directiory_list:
    info = wav.partition(".wav")[0].split("_")[1].replace(r"[0-9]", "")
    emotion = re.split(r"[0-9]", info)[0]
    if emotion=='a':
        emotion_df.append(("angry", Savee + "/" + wav))
    elif emotion=='d':
        emotion_df.append(("disgust", Savee + "/" + wav))
    elif emotion=='f':
        emotion_df.append(("fear", Savee + "/" + wav))
    elif emotion=='h':
        emotion_df.append(("happy", Savee + "/" + wav))
    elif emotion=='n':
        emotion_df.append(("neutral", Savee + "/" + wav))
    elif emotion=='sa':
        emotion_df.append(("sad", Savee + "/" + wav))
    else:
        emotion_df.append(("surprise", Savee + "/" + wav))


Savee_df = pd.DataFrame.from_dict(emotion_df)
Savee_df.rename(columns={1 : "Path", 0 : "Emotion"}, inplace=True)

Savee_df.head()

df = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)
df.shape

df.head(10)

df.drop(df.index[df['Emotion'] == 'surprise'], inplace=True)

"""**DISTRIBUTION OF DATA**"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

plt.style.use("ggplot")

plt.title("Count of emotions:")
sns.countplot(x=df["Emotion"])
sns.despine(top=True, right=True, left=False, bottom=False)

def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title(f'Waveplot for audio with {e} emotion', size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
    plt.savefig('plots/count_of_emotions.png')

def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.savefig(f'plots/spectrogram_{emotion}.png')

unique_emotions = df["Emotion"].unique()
for emotion in unique_emotions:
    file_path = df[df["Emotion"] == emotion]["Path"].iloc[0]
    try:
        data, sr = librosa.load(file_path, sr=None)
        create_waveplot(data, sr, emotion)
        create_spectrogram(data, sr, emotion)
    except Exception as e:
        print(f"Error processing file for emotion '{emotion}' at path '{file_path}': {e}")

""" **TEXT FEATURE EXTRACTION**"""


# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Load Whisper model
model = whisper.load_model("base")

"""SAVEE"""

savee_audio_folder = "datasets/Savee"
output_file = "transcriptions/SAVEE_transcriptions.txt"

# Dictionary to store transcriptions
transcriptions = {}

# Iterate through all files in the folder
for file_name in os.listdir(savee_audio_folder):
    if file_name.endswith((".wav", ".mp3", ".m4a")):  # Supported formats
        file_path = os.path.join(savee_audio_folder, file_name)  # Get the full file path
        print(f"Processing {file_name}...")

        try:
            # Transcribe the audio
            result = model.transcribe(file_path)
            transcriptions[file_name] = result["text"]
            print(f"Transcription for {file_name}:\n{result['text']}\n")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Save transcriptions to a file
with open(output_file, "w") as f:
    for file_name, transcription in transcriptions.items():
        f.write(f"{file_name}:\n")
        f.write(f"{transcription}\n\n")

print(f"Transcriptions saved to {output_file}")

def custom_tokenize(text):
    """
    Tokenize the text using regular expressions to split by non-alphanumeric characters.
    """
    return re.findall(r'\b\w+\b', text.lower())  # Extract words (alphanumeric only) and convert to lowercase

# Preprocessing function to clean the text (tokenize + remove stopwords)
def preprocess_text(text):
    """
    Preprocess the given text by tokenizing, converting to lowercase,
    and removing stopwords.
    """
    tokens = custom_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words("english"))  # Get list of English stopwords
    return [word for word in tokens if word not in stop_words]

# Load transcriptions from the output file
transcriptions_file = "transcriptions/SAVEE_transcriptions.txt"
transcriptions = {}

# Read transcriptions from the saved file
with open(transcriptions_file, "r") as f:
    lines = f.readlines()
    current_file = None
    for line in lines:
        if line.strip().endswith(".wav:"):  # Identify file names
            current_file = line.strip().replace(":", "")
            transcriptions[current_file] = ""
        elif current_file:
            transcriptions[current_file] += line.strip()

# Preprocess the transcriptions
processed_texts = {file_name: preprocess_text(text) for file_name, text in transcriptions.items()}

# Check the preprocessed output
for file_name, tokens in processed_texts.items():
    print(f"Processed Tokens for {file_name}: {tokens}")

# Train a Word2Vec model on the processed tokens
def train_word2vec(processed_texts, vector_size=100, window=5, min_count=1, workers=4):
    """
    Train a Word2Vec model on the processed tokens.
    """
    sentences = list(processed_texts.values())  # Extract token lists
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# Train the Word2Vec model
word2vec_model = train_word2vec(processed_texts)

# Generate feature vectors for each transcription
def generate_feature_vectors(processed_texts, model):
    """
    Generate feature vectors for each transcription by averaging the Word2Vec vectors of its tokens.
    """
    feature_vectors = {}
    for file_name, tokens in processed_texts.items():
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if vectors:
            feature_vectors[file_name] = np.mean(vectors, axis=0)  # Average the vectors
        else:
            feature_vectors[file_name] = np.zeros(model.vector_size)  # Fallback to zero vector
    return feature_vectors

# Generate the feature vectors
feature_vectors = generate_feature_vectors(processed_texts, word2vec_model)

# Save feature vectors to a file
output_feature_vectors_file = "feature_csv/SAVEE_feature_vectors.npy"
np.save(output_feature_vectors_file, feature_vectors)

print(f"Feature vectors successfully saved to {output_feature_vectors_file}")

# Check the feature vectors
for file_name, vector in feature_vectors.items():
    print(f"Feature Vector for {file_name}: {vector[:10]}... (truncated)")  # Print first 10 dimensions

# File paths
npy_file_path = "feature_csv/SAVEE_feature_vectors.npy"  # Saved .npy file
csv_file_path = "feature_csv/SAVEE_feature_vectors_with_labels.csv"  # CSV output path

# Load the feature vectors
feature_vectors = np.load(npy_file_path, allow_pickle=True).item()  # Load as dictionary

# Define a mapping from filename prefixes to emotion labels
emotion_mapping = {
    "a0": "angry", "a1": "angry",
    "d0": "disgust", "d1": "disgust",
    "f0": "fear", "f1": "fear",
    "h0": "happy", "h1": "happy",
    "n0": "neutral", "n1": "neutral", "n2": "neutral", "n3": "neutral",
    "sa": "sad",
    "su": "surprise"
}

# Extract features and labels
data = []
for file_name, vector in feature_vectors.items():
    emotion_code = os.path.basename(file_name).split("_")[1][0:2].lower()  # Extract first 1-2 letters after "_"
    emotion_label = emotion_mapping.get(emotion_code, "unknown")  # Map to emotion or 'unknown' if not found
    data.append([file_name, emotion_label] + list(vector))  # Combine filename, emotion, and feature vector

# Create a DataFrame
columns = ["File_Name", "Emotion"] + [f"Feature_{i}" for i in range(len(vector))]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(csv_file_path, index=False)

print(f"Feature vectors with labels successfully saved to {csv_file_path}")

# Quick check of the DataFrame
print(df.head())  # Display the first few rows of the DataFrame

"""RAVDESS"""

# Emotion mapping for RAVDESS
emotion_mapping_ravdess = {
    1: "neutral", 2: "neutral", 3: "happy", 4: "sad",
    5: "angry", 6: "fear", 7: "disgust", 8: "surprise"
}

# Set the base directory for RAVDESS dataset
base_dir = "datasets/Ravdess"

# Initialize a dictionary to store transcriptions and emotions
transcriptions = {}
emotion_data = []

# Traverse the RAVDESS folder structure
for i in range(1, 25):  # Actors are numbered from 1 to 24
    actor_folder = os.path.join(base_dir, f"Actor_{i:02d}")
    audio_files = glob.glob(os.path.join(actor_folder, "*.wav"))

    print(f"Found {len(audio_files)} files in {actor_folder}")
    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        try:
            # Extract emotion label from filename (3rd part of the name, e.g., "03" for happy)
            info = file_name.split("-")
            emotion_code = int(info[2])  # 3rd field is the emotion code
            emotion_label = emotion_mapping_ravdess.get(emotion_code, "unknown")

            # Transcribe audio
            print(f"Processing {file_name}...")
            result = model.transcribe(file_path)
            transcription = result["text"]

            # Store transcription and emotion
            transcriptions[file_name] = transcription
            emotion_data.append((file_name, emotion_label, transcription))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Convert emotion data to a DataFrame
emotion_df = pd.DataFrame(emotion_data, columns=["File_Name", "Emotion", "Transcription"])

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalnum() and word not in stop_words]

processed_texts = {
    row["File_Name"]: preprocess_text(row["Transcription"])
    for _, row in emotion_df.iterrows()
}

# Train Word2Vec model on processed texts
word2vec_model = Word2Vec(
    list(processed_texts.values()),  # Tokenized transcriptions
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Generate feature vectors
def generate_feature_vectors(processed_texts, word2vec_model):
    feature_vectors = []

    for tokens in processed_texts.values():
        vector = np.zeros(word2vec_model.vector_size)
        count = 0
        for token in tokens:
            if token in word2vec_model.wv:
                vector += word2vec_model.wv[token]
                count += 1
        if count > 0:
            vector /= count  # Average the token vectors
        feature_vectors.append(vector)

    return np.array(feature_vectors)

feature_vectors = generate_feature_vectors(processed_texts, word2vec_model)

# Combine emotion labels and feature vectors into a DataFrame
feature_df = pd.DataFrame(feature_vectors)
feature_df.columns = [f"Feature_{i}" for i in range(feature_df.shape[1])]
feature_df["File_Name"] = emotion_df["File_Name"]
feature_df["Emotion"] = emotion_df["Emotion"]

# Save the combined DataFrame to CSV
csv_file_path = "feature_csv/ravdess_features_emotions.csv"
feature_df.to_csv(csv_file_path, index=False)

print(f"Feature vectors and emotions successfully saved to {csv_file_path}")

"""CREMA-D"""

# Emotion mapping for CREMA-D
emotion_mapping_crema = {
    "SAD": "sad",
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SUR": "surprise"
}

# Set the base directory for CREMA-D dataset
crema_audio_dir = "datasets/Crema"

# Initialize a list to store data
emotion_data = []
transcriptions = {}

# Process audio files in CREMA-D
for file_name in os.listdir(crema_audio_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(crema_audio_dir, file_name)

        # Extract emotion label from the filename
        try:
            emotion_code = file_name.split("_")[2]
            emotion_label = emotion_mapping_crema.get(emotion_code, "unknown")

            # Transcribe the audio
            print(f"Processing {file_name}...")
            result = model.transcribe(file_path)
            transcription = result["text"]

            # Append data
            transcriptions[file_name] = transcription
            emotion_data.append((file_name, emotion_label, transcription))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Convert to DataFrame
emotion_df = pd.DataFrame(emotion_data, columns=["File_Name", "Emotion", "Transcription"])

# Preprocess transcriptions
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalnum() and word not in stop_words]

processed_texts = {
    row["File_Name"]: preprocess_text(row["Transcription"])
    for _, row in emotion_df.iterrows()
}

# Train Word2Vec model on processed texts
word2vec_model = Word2Vec(
    list(processed_texts.values()),  # Tokenized transcriptions
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Generate feature vectors
def generate_feature_vectors(processed_texts, word2vec_model):
    feature_vectors = []

    for tokens in processed_texts.values():
        vector = np.zeros(word2vec_model.vector_size)
        count = 0
        for token in tokens:
            if token in word2vec_model.wv:
                vector += word2vec_model.wv[token]
                count += 1
        if count > 0:
            vector /= count  # Average the token vectors
        feature_vectors.append(vector)

    return np.array(feature_vectors)

feature_vectors = generate_feature_vectors(processed_texts, word2vec_model)

# Combine emotion labels and feature vectors into a DataFrame
feature_df = pd.DataFrame(feature_vectors)
feature_df.columns = [f"Feature_{i}" for i in range(feature_df.shape[1])]
feature_df["File_Name"] = emotion_df["File_Name"]
feature_df["Emotion"] = emotion_df["Emotion"]

# Save the combined DataFrame to CSV
csv_file_path = "feature_csv/crema_features_emotions.csv"
feature_df.to_csv(csv_file_path, index=False)

print(f"Feature vectors and emotions successfully saved to {csv_file_path}")

"""TESS"""

# Emotion mapping for TESS (based on file names)
emotion_mapping_tess = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "surprise",
    "sad": "sad"
}

# Set the base directory for TESS dataset
tess_audio_dir = "datasets/Tess"

# Initialize a list to store data
emotion_data = []
transcriptions = {}

# Process audio files in TESS
for file_name in os.listdir(tess_audio_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(tess_audio_dir, file_name)

        # Extract emotion label from the filename
        try:
            emotion_code = file_name.split("_")[2].replace(".wav", "")
            emotion_label = emotion_mapping_tess.get(emotion_code, "unknown")

            # Transcribe the audio
            print(f"Processing {file_name}...")
            result = model.transcribe(file_path)
            transcription = result["text"]

            # Append data
            transcriptions[file_name] = transcription
            emotion_data.append((file_name, emotion_label, transcription))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Convert to DataFrame
emotion_df = pd.DataFrame(emotion_data, columns=["File_Name", "Emotion", "Transcription"])

# Preprocess transcriptions
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word.isalnum() and word not in stop_words]

processed_texts = {
    row["File_Name"]: preprocess_text(row["Transcription"])
    for _, row in emotion_df.iterrows()
}

# Train Word2Vec model on processed texts
word2vec_model = Word2Vec(
    list(processed_texts.values()),  # Tokenized transcriptions
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

# Generate feature vectors
def generate_feature_vectors(processed_texts, word2vec_model):
    feature_vectors = []

    for tokens in processed_texts.values():
        vector = np.zeros(word2vec_model.vector_size)
        count = 0
        for token in tokens:
            if token in word2vec_model.wv:
                vector += word2vec_model.wv[token]
                count += 1
        if count > 0:
            vector /= count  # Average the token vectors
        feature_vectors.append(vector)

    return np.array(feature_vectors)

feature_vectors = generate_feature_vectors(processed_texts, word2vec_model)

# Combine emotion labels and feature vectors into a DataFrame
feature_df = pd.DataFrame(feature_vectors)
feature_df.columns = [f"Feature_{i}" for i in range(feature_df.shape[1])]
feature_df["File_Name"] = emotion_df["File_Name"]
feature_df["Emotion"] = emotion_df["Emotion"]

# Save the combined DataFrame to CSV
csv_file_path = "feature_csv/tess_features_emotions.csv"
feature_df.to_csv(csv_file_path, index=False)

print(f"Feature vectors and emotions successfully saved to {csv_file_path}")

"""**Exploring SVM Performance Across Different Datasets**

Combining Audio and Text Features from RAVDESS
"""

primary_csv_path = "feature_csv/ravdess_features_with_filenames.csv"  # CSV of Audio Features
secondary_csv_path = "feature_csv/ravdess_features_emotions.csv"  # CSV of Text Features
output_csv_path = "feature_csv/final_ravdess_combined_features.csv"

primary_df = pd.read_csv(primary_csv_path)
secondary_df = pd.read_csv(secondary_csv_path)

print(f"Primary CSV shape: {primary_df.shape}")
print(f"Secondary CSV shape: {secondary_df.shape}")

if "File_Name" not in primary_df.columns or "File_Name" not in secondary_df.columns:
    raise ValueError("Both CSVs must have a 'File_Name' column for matching.")

if "Emotion" not in primary_df.columns or "Emotion" not in secondary_df.columns:
    raise ValueError("Both CSVs must have an 'Emotion' column for matching.")

primary_df = primary_df.rename(columns=lambda col: f"Audio_{col}" if col.startswith("Feature_") else col)
secondary_df = secondary_df.rename(columns=lambda col: f"Text_{col}" if col.startswith("Feature_") else col)

combined_features = []
files_matched = 0

for idx, row in primary_df.iterrows():
    file_name = row["File_Name"]
    emotion_primary = row["Emotion"]

    matching_row = secondary_df[secondary_df["File_Name"] == file_name]

    if not matching_row.empty:
        matching_row = matching_row.iloc[0]
        emotion_secondary = matching_row["Emotion"]

        if emotion_primary == emotion_secondary:
            files_matched += 1

            combined_row = {
                "File_Name": file_name,
                "Emotion": emotion_primary,
            }

            audio_features = row.filter(like="Audio_").to_dict()

            text_features = matching_row.filter(like="Text_").to_dict()

            combined_row.update(audio_features)
            combined_row.update(text_features)

            combined_features.append(combined_row)

# Convert combined features to a DataFrame
combined_df = pd.DataFrame(combined_features)


print(f"Total files matched: {files_matched}")
print(f"Final combined DataFrame shape: {combined_df.shape}")

combined_df.to_csv(output_csv_path, index=False)
print(f"Combined features saved to {output_csv_path}")

"""Using all the 4 datasets"""

# File paths
ravdess_path = "feature_csv/final_ravdess_combined_features.csv"
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


combined_features = []

ravdess_df = pd.read_csv(ravdess_path)

# Correct Renaming of RAVDESS Columns (Due to Mismatched File Order in Audio and Text Files)
def rename_ravdess_columns(col):
    if col.startswith("Audio_Feature_"):
        return f"Feature_{col[len('Audio_Feature_'):]}_audio"
    elif col.startswith("Text_Feature_"):
        return f"Feature_{col[len('Text_Feature_'):]}_text"
    else:
        return col

ravdess_df.rename(columns=rename_ravdess_columns, inplace=True)

# Debugging
print("\nRAVDESS DataFrame after renaming:")
print(ravdess_df.head())


ravdess_audio_features = [col for col in ravdess_df.columns if col.endswith('_audio')]
ravdess_text_features = [col for col in ravdess_df.columns if col.endswith('_text')]
print(f"\nRAVDESS Dataset:")
print(f"Total audio features: {len(ravdess_audio_features)}")
print(f"Total text features: {len(ravdess_text_features)}")

# Add RAVDESS data to the combined list
combined_features.extend(ravdess_df.to_dict(orient="records"))

## Process SAVEE, TESS, CREMA
for audio_file, text_file in zip(audio_feature_files, text_feature_files):

    audio_df = pd.read_csv(audio_file)
    text_df = pd.read_csv(text_file)

    if 'File_Name' not in audio_df.columns:
        if 'File_Name' in text_df.columns and len(audio_df) == len(text_df):
            audio_df['File_Name'] = text_df['File_Name']
            print(f"\nAdded 'File_Name' to audio_df from text_df for {audio_file.split('/')[-1]}")
        else:
            print(f"\nWarning: 'File_Name' not found in audio_df or row counts do not match for {audio_file.split('/')[-1]}. Skipping this dataset.")
            continue

    if 'Emotion' not in audio_df.columns:
        print(f"\nWarning: 'Emotion' not found in audio_df for {audio_file.split('/')[-1]}. Skipping this dataset.")
        continue

    audio_df.rename(
        columns=lambda col: f"{col}_audio" if col.startswith("Feature_") and not col.endswith("_audio") else col,
        inplace=True
    )

    text_df.rename(
        columns=lambda col: f"{col}_text" if col.startswith("Feature_") and not col.endswith("_text") else col,
        inplace=True
    )

    print(f"\nAudio DataFrame from {audio_file.split('/')[-1]} after renaming:")
    print(audio_df.head())
    print(f"\nText DataFrame from {text_file.split('/')[-1]} after renaming:")
    print(text_df.head())

    dataset_name = audio_file.split('/')[-1].split('_')[0].upper()
    audio_features = [col for col in audio_df.columns if col.endswith('_audio')]
    text_features = [col for col in text_df.columns if col.endswith('_text')]
    print(f"\n{dataset_name} Dataset:")
    print(f"Total audio features: {len(audio_features)}")
    print(f"Total text features: {len(text_features)}")


    try:
        merged_df = pd.merge(audio_df, text_df, on=["File_Name", "Emotion"], how="inner")
        print(f"\nNumber of merged records for {dataset_name}: {merged_df.shape[0]}")

        combined_features.extend(merged_df.to_dict(orient="records"))
    except KeyError as e:
        print(f"\nKeyError while merging {dataset_name}: {e}")
        print("Skipping this dataset.")
        continue

combined_df = pd.DataFrame(combined_features)

print("\nFinal Combined DataFrame Preview:")
print(combined_df.head())
print(f"\nFinal Combined DataFrame Shape: {combined_df.shape}")

nan_counts = combined_df.isna().sum()
print("\nNaN Values in Each Column:")
print(nan_counts[nan_counts > 0])

"""SIMPLE SVM"""

print(combined_df.head())
X = combined_df.drop(['File_Name', 'Emotion'], axis=1)


y = combined_df['Emotion']

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Encoded Labels:", np.unique(y_encoded))
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])
svm_classifier = SVC(random_state=42)

svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy for SVM: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

"""Best Fit SVM"""

# This code can be executed with high computing power to achieve better accuracy.

# 1. Prepare Features and Target
X = combined_df.drop(['File_Name', 'Emotion'], axis=1)
y = combined_df['Emotion']

# 2. Encode Target Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Encoded Labels:", np.unique(y_encoded))
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape: {X_test.shape}")

# 4. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. SVM Pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

# 6. Hyperparameter Grid for SVM
param_grid_svm = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.001, 0.0001],
    'svm__kernel': ['rbf', 'linear', 'poly']
}

# 7. GridSearchCV for SVM
grid_search_svm = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid_svm,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# 8. Train SVM
grid_search_svm.fit(X_train, y_train)

# 9. Best Parameters and Score
print("Best Parameters for SVM:", grid_search_svm.best_params_)
print("Best Cross-validation Accuracy for SVM:", grid_search_svm.best_score_)

# 10. Predict on Test Set
y_pred_svm = grid_search_svm.predict(X_test)

# 11. Evaluate SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"\nTest Set Accuracy for SVM: {accuracy_svm:.4f}")

print("\nClassification Report for SVM:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))

print("\nConfusion Matrix for SVM:")
print(confusion_matrix(y_test, y_pred_svm))

# 12. Dimensionality Reduction with PCA
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('svm', SVC(random_state=42))
])

param_grid_pca = {
    'pca__n_components': [0.95, 0.99],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto', 0.001],
    'svm__kernel': ['rbf', 'linear']
}

grid_search_pca = GridSearchCV(
    estimator=pca_pipeline,
    param_grid=param_grid_pca,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# 13. Train PCA + SVM
grid_search_pca.fit(X_train, y_train)

# 14. Best Parameters and Score for PCA + SVM
print("Best Parameters with PCA:", grid_search_pca.best_params_)
print("Best Cross-validation Accuracy with PCA:", grid_search_pca.best_score_)

# 15. Predict on Test Set with PCA
y_pred_pca = grid_search_pca.predict(X_test)

# 16. Evaluate PCA + SVM
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"\nTest Set Accuracy with PCA + SVM: {accuracy_pca:.4f}")

print("\nClassification Report with PCA + SVM:")
print(classification_report(y_test, y_pred_pca, target_names=label_encoder.classes_))

print("\nConfusion Matrix with PCA + SVM:")
print(confusion_matrix(y_test, y_pred_pca))

# 17. Alternative Classifier: Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

# 18. Train Random Forest
grid_search_rf.fit(X_train, y_train)

# 19. Best Parameters and Score for Random Forest
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Cross-validation Accuracy for Random Forest:", grid_search_rf.best_score_)

# 20. Predict on Test Set with Random Forest
y_pred_rf = grid_search_rf.predict(X_test)

# 21. Evaluate Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nTest Set Accuracy for Random Forest: {accuracy_rf:.4f}")

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

print("\nConfusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

# 22. Ensemble Method: Voting Classifier
svm_clf = SVC(C=1, gamma='scale', kernel='linear', probability=True, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('svm', svm_clf), ('rf', rf_clf)],
    voting='soft'
)

# Train Voting Classifier
voting_clf.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred_voting = voting_clf.predict(X_test_scaled)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print(f"\nTest Set Accuracy for Voting Classifier: {accuracy_voting:.4f}")

print("\nClassification Report for Voting Classifier:")
print(classification_report(y_test, y_pred_voting, target_names=label_encoder.classes_))

print("\nConfusion Matrix for Voting Classifier:")
print(confusion_matrix(y_test, y_pred_voting))