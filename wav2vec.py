"""#**SAVEE TRANSFORMATION FEATURE EXTRACTION**"""
import os
import re
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

# Path to your Savee dataset directory
Savee = 'datasets/Savee'

# List all .wav files in the directory
savee_directory_list = [f for f in os.listdir(Savee) if f.lower().endswith('.wav')]

emotion_df = []

for wav in savee_directory_list:
    # Extract the part before '.wav' and split by '_'
    info = wav.partition(".wav")[0].split("_")[1]

    # Use regex to remove digits
    emotion_code = re.sub(r"\d", "", info).lower()

    # Map emotion codes to emotion labels
    if emotion_code == 'a':
        emotion = "angry"
    elif emotion_code == 'd':
        emotion = "disgust"
    elif emotion_code == 'f':
        emotion = "fear"
    elif emotion_code == 'h':
        emotion = "happy"
    elif emotion_code == 'n':
        emotion = "neutral"
    elif emotion_code == 'sa':
        emotion = "sad"
    else:
        emotion = "surprise"

    # Append a tuple of (Emotion, Path)
    emotion_df.append((emotion, os.path.join(Savee, wav)))

# Create DataFrame with proper column names
Savee_df = pd.DataFrame(emotion_df, columns=["Emotion", "Path"])

# Display the first few rows to verify
print("Dataset preview:")
print(Savee_df.head())
print("Columns:", Savee_df.columns)
print(f"Total files: {len(Savee_df)}")

# Load Wav2Vec 2.0 model and processor
def load_wav2vec_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model.eval()  # Set to evaluation mode
        print("Wav2Vec 2.0 model loaded successfully.")
    except Exception as e:
        print(f"Error loading Wav2Vec 2.0 model: {e}")
        raise e
    return processor, model, device

# Extract Wav2Vec features for a single audio file
def extract_wav2vec_features(audio_path, processor, model, device):
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None

        # Load audio at 16kHz (required for Wav2Vec 2.0)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Preprocess audio for Wav2Vec 2.0
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        # Move inputs to device and ensure float32 precision
        inputs = {key: tensor.to(device).type(torch.float32) for key, tensor in inputs.items()}

        # Ensure model is in float32
        model = model.to(torch.float32)

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state  # Shape: (1, sequence_length, feature_dim)

        # Aggregate features (mean-pooling across time dimension)
        pooled_features = features.mean(dim=1).squeeze().cpu().numpy()  # Shape: (feature_dim,)

        return pooled_features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Process entire dataset to extract Wav2Vec features
def process_wav2vec_dataset_with_filenames(df, processor, model, device):
    wav2vec_features = []
    emotions = []
    file_names = []
    for path, emotion in tqdm(zip(df.Path, df.Emotion), total=len(df), desc="Extracting features"):
        file_name = os.path.basename(path)  # Extract file name
        features = extract_wav2vec_features(path, processor, model, device)
        if features is not None:
            wav2vec_features.append(features)
            emotions.append(emotion)
            file_names.append(file_name)
        else:
            print(f"Skipping file due to error: {path}")
    print(f"Total features extracted: {len(wav2vec_features)}")
    print(f"Total emotions recorded: {len(emotions)}")
    print(f"Total file names recorded: {len(file_names)}")
    return np.array(wav2vec_features), emotions, file_names

# Save features to a CSV file
def save_features_to_csv(features, emotions, file_names, output_filename):
    if len(features) != len(emotions) or len(features) != len(file_names):
        print("Mismatch between features, emotions, and file names. Check the processing step.")
        print(f"Features count: {len(features)}, Emotions count: {len(emotions)}, File names count: {len(file_names)}")
        return
    print("Sample file names:", file_names[:5])

    feature_columns = [f"Feature_{i+1}" for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)
    emotions_df = pd.Series(emotions, name="Emotion")
    file_names_df = pd.Series(file_names, name="File_Name")
    df = pd.concat([features_df, emotions_df, file_names_df], axis=1)

    # Debugging: Check the DataFrame before saving
    print("DataFrame preview before saving:")
    print(df.head())
    print("Columns:", df.columns)

    df.to_csv(output_filename, index=False)
    print(f"Features, emotions, and file names saved to {output_filename}")

# Main Workflow
if __name__ == "__main__":
    # Load Wav2Vec 2.0 model, processor, and device
    processor, model, device = load_wav2vec_model()

    # Process the dataset to extract features
    wav2vec_features, emotions, file_names = process_wav2vec_dataset_with_filenames(Savee_df, processor, model, device)
    print(f"Feature extraction complete. Shape: {wav2vec_features.shape}")

    # Save features to CSV
    output_file = "feature_csv/saveetransformer_features_with_filenames.csv"
    save_features_to_csv(wav2vec_features, emotions, file_names, output_file)

# manual and transformer feature merging
import pandas as pd

# Paths to the manual and transformer feature CSV files
manual_features_path = "feature_csv/manual_features_with_filenames_final.csv"
transformer_features_path = "feature_csv/saveetransformer_features_with_filenames.csv"

# Output path for the merged CSV
merged_features_path = "feature_csv/merged_manual_transformer_savee_features.csv"

# Load the manual and transformer feature CSVs into DataFrames
manual_df = pd.read_csv(manual_features_path)
transformer_df = pd.read_csv(transformer_features_path)

# Ensure both DataFrames have a File_Name column
if "File_Name" not in manual_df.columns or "File_Name" not in transformer_df.columns:
    raise ValueError("Both CSV files must have a 'File_Name' column for merging.")

# Merge the DataFrames on the 'File_Name' column
merged_df = pd.merge(manual_df, transformer_df, on="File_Name", suffixes=("_manual", "_transformer"))

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(merged_features_path, index=False)

print(f"Merged features saved to {merged_features_path}")
print("Merged DataFrame preview:")
print(merged_df.head())

"""# **RAVEDESS TRANSFORMER FEATURE EXTRACTION**"""

# Path to your RAVDESS dataset directory
Ravdess = 'datasets/Ravdess'

# List all directories (each actor's folder) in RAVDESS
ravdess_directory_list = [d for d in os.listdir(Ravdess) if os.path.isdir(os.path.join(Ravdess, d))]

emotion_df = []

# Traverse each actor's directory
for actor_dir in ravdess_directory_list:
    actor_files = os.listdir(os.path.join(Ravdess, actor_dir))  # List all files for each actor
    for wav in actor_files:
        # Check if the file is a .wav file and follows the expected naming pattern
        if wav.endswith(".wav"):
            info = wav.partition(".wav")[0].split("-")
            if len(info) > 2:  # Ensure there are at least 3 parts
                emotion_code = int(info[2])  # Extract emotion code as an integer
                # Map emotion code to emotion labels
                emotion_map = {
                    1: 'neutral',
                    2: 'neutral',
                    3: 'happy',
                    4: 'sad',
                    5: 'angry',
                    6: 'fear',
                    7: 'disgust',
                    8: 'surprise'
                }
                emotion = emotion_map.get(emotion_code, "unknown")  # Map code to label
                # Append emotion, file name, and full path to the .wav file
                emotion_df.append((emotion, wav, os.path.join(Ravdess, actor_dir, wav)))

# Create DataFrame with proper column names
Ravdess_df = pd.DataFrame(emotion_df, columns=["Emotion", "File_Name", "Path"])

# Display the first few rows to verify
print("Dataset preview:")
print(Ravdess_df.head())
print("Columns:", Ravdess_df.columns)
print(f"Total files: {len(Ravdess_df)}")

# Load Wav2Vec 2.0 model and processor
def load_wav2vec_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model.eval()  # Set to evaluation mode
        print("Wav2Vec 2.0 model loaded successfully.")
    except Exception as e:
        print(f"Error loading Wav2Vec 2.0 model: {e}")
        raise e
    return processor, model, device

# Extract Wav2Vec features for a single audio file
def extract_wav2vec_features(audio_path, processor, model, device):
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None

        # Load audio at 16kHz (required for Wav2Vec 2.0)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Preprocess audio for Wav2Vec 2.0
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        # Move inputs to device and ensure float32 precision
        inputs = {key: tensor.to(device).type(torch.float32) for key, tensor in inputs.items()}

        # Ensure model is in float32
        model = model.to(torch.float32)

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state  # Shape: (1, sequence_length, feature_dim)

        # Aggregate features (mean-pooling across time dimension)
        pooled_features = features.mean(dim=1).squeeze().cpu().numpy()  # Shape: (feature_dim,)

        return pooled_features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Process entire dataset to extract Wav2Vec features
def process_wav2vec_dataset(df, processor, model, device):
    file_names = []
    wav2vec_features = []
    emotions = []

    for path, file_name, emotion in tqdm(zip(df.Path, df.File_Name, df.Emotion), total=len(df), desc="Extracting features"):
        features = extract_wav2vec_features(path, processor, model, device)
        if features is not None:
            wav2vec_features.append(features)
            emotions.append(emotion)
            file_names.append(file_name)  # Store the file name
        else:
            print(f"Skipping file due to error: {path}")

    print(f"Total features extracted: {len(wav2vec_features)}")
    print(f"Total emotions recorded: {len(emotions)}")
    print(f"Total file names recorded: {len(file_names)}")
    return file_names, np.array(wav2vec_features), emotions

# Save features to a CSV file
def save_features_to_csv(file_names, features, emotions, output_filename):
    if len(features) != len(emotions) or len(file_names) != len(features):
        print("Mismatch in lengths of file names, features, and emotions. Check the processing step.")
        print(f"File names count: {len(file_names)}, Features count: {len(features)}, Emotions count: {len(emotions)}")
        return

    # Prepare DataFrame
    feature_columns = [f"Feature_{i+1}" for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)
    file_names_df = pd.Series(file_names, name="File_Name")
    emotions_df = pd.Series(emotions, name="Emotion")

    # Combine all into a single DataFrame
    df = pd.concat([file_names_df, emotions_df, features_df], axis=1)

    # Debugging: Check the DataFrame before saving
    print("DataFrame preview before saving:")
    print(df.head())
    print("Columns:", df.columns)

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"File names, features, and emotions saved to {output_filename}")

# Main Workflow
if __name__ == "__main__":
    # Load Wav2Vec 2.0 model, processor, and device
    processor, model, device = load_wav2vec_model()

    # Process the RAVDESS dataset to extract features
    file_names, wav2vec_features, emotions = process_wav2vec_dataset(Ravdess_df, processor, model, device)
    print(f"Feature extraction complete. Shape: {wav2vec_features.shape}")

    # Save features, file names, and emotions to CSV
    output_file = "feature_csv/ravdess_features_with_filenames.csv"
    save_features_to_csv(file_names, wav2vec_features, emotions, output_file)

"""# **CREMA TRANSFORMER FEATURE EXTRACTION**"""

# Path to the CREMA-D dataset directory
Crema = "datasets/Crema"

# Create an emotion mapping based on the CREMA-D file naming convention
emotion_df = []
for wav in os.listdir(Crema):
    if wav.endswith(".wav"):  # Process only .wav files
        info = wav.partition(".wav")[0].split("_")
        # Map emotion codes to emotion labels
        if info[2] == 'SAD':
            emotion_df.append(("sad", wav, os.path.join(Crema, wav)))
        elif info[2] == 'ANG':
            emotion_df.append(("angry", wav, os.path.join(Crema, wav)))
        elif info[2] == 'DIS':
            emotion_df.append(("disgust", wav, os.path.join(Crema, wav)))
        elif info[2] == 'FEA':
            emotion_df.append(("fear", wav, os.path.join(Crema, wav)))
        elif info[2] == 'HAP':
            emotion_df.append(("happy", wav, os.path.join(Crema, wav)))
        elif info[2] == 'NEU':
            emotion_df.append(("neutral", wav, os.path.join(Crema, wav)))
        else:
            emotion_df.append(("surprise", wav, os.path.join(Crema, wav)))

# Create DataFrame with proper column names
Crema_df = pd.DataFrame(emotion_df, columns=["Emotion", "File_Name", "Path"])

# Display the first few rows to verify
print("Dataset preview:")
print(Crema_df.head())
print("Columns:", Crema_df.columns)
print(f"Total files: {len(Crema_df)}")

# Load Wav2Vec 2.0 model and processor
def load_wav2vec_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model.eval()  # Set to evaluation mode
        print("Wav2Vec 2.0 model loaded successfully.")
    except Exception as e:
        print(f"Error loading Wav2Vec 2.0 model: {e}")
        raise e
    return processor, model, device

# Extract Wav2Vec features for a single audio file
def extract_wav2vec_features(audio_path, processor, model, device):
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None

        # Load audio at 16kHz (required for Wav2Vec 2.0)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Preprocess audio for Wav2Vec 2.0
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        # Move inputs to device and ensure float32 precision
        inputs = {key: tensor.to(device).type(torch.float32) for key, tensor in inputs.items()}

        # Ensure model is in float32
        model = model.to(torch.float32)

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state  # Shape: (1, sequence_length, feature_dim)

        # Aggregate features (mean-pooling across time dimension)
        pooled_features = features.mean(dim=1).squeeze().cpu().numpy()  # Shape: (feature_dim,)

        return pooled_features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Process entire dataset to extract Wav2Vec features
def process_wav2vec_dataset(df, processor, model, device):
    file_names = []
    wav2vec_features = []
    emotions = []

    for path, file_name, emotion in tqdm(zip(df.Path, df.File_Name, df.Emotion), total=len(df), desc="Extracting features"):
        features = extract_wav2vec_features(path, processor, model, device)
        if features is not None:
            wav2vec_features.append(features)
            emotions.append(emotion)
            file_names.append(file_name)  # Store the file name
        else:
            print(f"Skipping file due to error: {path}")

    print(f"Total features extracted: {len(wav2vec_features)}")
    print(f"Total emotions recorded: {len(emotions)}")
    print(f"Total file names recorded: {len(file_names)}")
    return file_names, np.array(wav2vec_features), emotions

# Save features to a CSV file
def save_features_to_csv(file_names, features, emotions, output_filename):
    if len(features) != len(emotions) or len(file_names) != len(features):
        print("Mismatch in lengths of file names, features, and emotions. Check the processing step.")
        print(f"File names count: {len(file_names)}, Features count: {len(features)}, Emotions count: {len(emotions)}")
        return

    # Prepare DataFrame
    feature_columns = [f"Feature_{i+1}" for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)
    file_names_df = pd.Series(file_names, name="File_Name")
    emotions_df = pd.Series(emotions, name="Emotion")

    # Combine all into a single DataFrame
    df = pd.concat([file_names_df, emotions_df, features_df], axis=1)

    # Debugging: Check the DataFrame before saving
    print("DataFrame preview before saving:")
    print(df.head())
    print("Columns:", df.columns)

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"File names, features, and emotions saved to {output_filename}")

# Main Workflow
if __name__ == "__main__":
    # Load Wav2Vec 2.0 model, processor, and device
    processor, model, device = load_wav2vec_model()

    # Process the CREMA-D dataset to extract features
    file_names, wav2vec_features, emotions = process_wav2vec_dataset(Crema_df, processor, model, device)
    print(f"Feature extraction complete. Shape: {wav2vec_features.shape}")

    # Save features, file names, and emotions to CSV
    output_file = "feature_csv/crema_features_audio.csv"
    save_features_to_csv(file_names, wav2vec_features, emotions, output_file)

"""# **TESS TRANSFORMER FEATURE EXTRACTION**"""

# Path to the TESS dataset directory
Tess = "datasets/Tess"

# Parse the TESS dataset to create a DataFrame with Emotion and Path
emotion_df = []

# Process .wav files in the TESS directory
for wav in os.listdir(Tess):
    if wav.endswith(".wav"):  # Process only .wav files
        info = wav.partition(".wav")[0].split("_")
        emo = info[2]  # Extract emotion part from the filename
        emotion_label = "surprise" if emo == "ps" else emo
        emotion_df.append((emotion_label, wav, os.path.join(Tess, wav)))

# Create DataFrame with proper column names
Tess_df = pd.DataFrame(emotion_df, columns=["Emotion", "File_Name", "Path"])

# Display the first few rows to verify
print("Dataset preview:")
print(Tess_df.head())
print("Columns:", Tess_df.columns)
print(f"Total files: {len(Tess_df)}")

# Load Wav2Vec 2.0 model and processor
def load_wav2vec_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        model.eval()  # Set to evaluation mode
        print("Wav2Vec 2.0 model loaded successfully.")
    except Exception as e:
        print(f"Error loading Wav2Vec 2.0 model: {e}")
        raise e
    return processor, model, device

# Extract Wav2Vec features for a single audio file
def extract_wav2vec_features(audio_path, processor, model, device):
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            return None

        # Load audio at 16kHz (required for Wav2Vec 2.0)
        audio, sr = librosa.load(audio_path, sr=16000)

        # Preprocess audio for Wav2Vec 2.0
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        # Move inputs to device and ensure float32 precision
        inputs = {key: tensor.to(device).type(torch.float32) for key, tensor in inputs.items()}

        # Ensure model is in float32
        model = model.to(torch.float32)

        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state  # Shape: (1, sequence_length, feature_dim)

        # Aggregate features (mean-pooling across time dimension)
        pooled_features = features.mean(dim=1).squeeze().cpu().numpy()  # Shape: (feature_dim,)

        return pooled_features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Process entire dataset to extract Wav2Vec features
def process_wav2vec_dataset(df, processor, model, device):
    file_names = []
    wav2vec_features = []
    emotions = []

    for path, file_name, emotion in tqdm(zip(df.Path, df.File_Name, df.Emotion), total=len(df), desc="Extracting features"):
        features = extract_wav2vec_features(path, processor, model, device)
        if features is not None:
            wav2vec_features.append(features)
            emotions.append(emotion)
            file_names.append(file_name)  # Store the file name
        else:
            print(f"Skipping file due to error: {path}")

    print(f"Total features extracted: {len(wav2vec_features)}")
    print(f"Total emotions recorded: {len(emotions)}")
    print(f"Total file names recorded: {len(file_names)}")
    return file_names, np.array(wav2vec_features), emotions

# Save features to a CSV file
def save_features_to_csv(file_names, features, emotions, output_filename):
    if len(features) != len(emotions) or len(file_names) != len(features):
        print("Mismatch in lengths of file names, features, and emotions. Check the processing step.")
        print(f"File names count: {len(file_names)}, Features count: {len(features)}, Emotions count: {len(emotions)}")
        return

    # Prepare DataFrame
    feature_columns = [f"Feature_{i+1}" for i in range(features.shape[1])]
    features_df = pd.DataFrame(features, columns=feature_columns)
    file_names_df = pd.Series(file_names, name="File_Name")
    emotions_df = pd.Series(emotions, name="Emotion")

    # Combine all into a single DataFrame
    df = pd.concat([file_names_df, emotions_df, features_df], axis=1)

    # Debugging: Check the DataFrame before saving
    print("DataFrame preview before saving:")
    print(df.head())
    print("Columns:", df.columns)

    # Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"File names, features, and emotions saved to {output_filename}")

# Main Workflow
if __name__ == "__main__":
    # Load Wav2Vec 2.0 model, processor, and device
    processor, model, device = load_wav2vec_model()

    # Process the TESS dataset to extract features
    file_names, wav2vec_features, emotions = process_wav2vec_dataset(Tess_df, processor, model, device)
    print(f"Feature extraction complete. Shape: {wav2vec_features.shape}")

    # Save features, file names, and emotions to CSV
    output_file = "feature_csv/tess_features_audio.csv"
    save_features_to_csv(file_names, wav2vec_features, emotions, output_file)