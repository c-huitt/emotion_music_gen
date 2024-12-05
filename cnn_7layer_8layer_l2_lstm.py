import os
import re
from glob import glob
import itertools
import warnings
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import entropy
from keras import utils, layers, models, backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""DATASET PREPROCESSING
**RAVDESS**
"""

Ravdess = "datasets/Ravdess"
base_dir = "datasets/Ravdess"

audio_extensions = ["*.wav"]

audio_files = []

for i in range(1, 25):  # Loop from 1 to 24
    actor_folder = os.path.join(base_dir, f"Actor_{i:02d}")
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(actor_folder, ext)))

print(f"Total number of audio files: {len(audio_files)}")

for filename in os.listdir(Ravdess):
    print(filename)

ravdess_directory_list = os.listdir(Ravdess)
emotion_df = []

for dir in ravdess_directory_list:
    actor = os.listdir(os.path.join(Ravdess, dir))
    for wav in actor:
        if wav.endswith(".wav"):
            info = wav.partition(".wav")[0].split("-")
            if len(info) > 2:
                emotion = int(info[2])
                emotion_df.append((emotion, os.path.join(Ravdess, dir, wav)))

print(emotion_df)

Ravdess_df = pd.DataFrame.from_dict(emotion_df)
Ravdess_df.rename(columns={1 : "Path", 0 : "Emotion"}, inplace=True)

Ravdess_df.Emotion.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()

"""**CREMA DATASET**"""

Crema = "datasets/Crema"

audio_dir = "datasets/Crema"

audio_extensions = ["*.wav"]

audio_files = []
for ext in audio_extensions:
    audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))

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

audio_dir = "datasets/Tess"

audio_extensions = ["*.wav"]


audio_files = []
for ext in audio_extensions:
    audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))

print(f"Total number of audio files: {len(audio_files)}")

for filename in os.listdir(Tess):
    print(filename)

emotion_df = []

for wav in os.listdir(Tess):
    if wav.endswith(".wav"):
        info = wav.partition(".wav")[0].split("_")
        emo = info[2]
        emotion_label = "surprise" if emo == "ps" else emo
        emotion_df.append((emotion_label, os.path.join(Tess, wav)))

Tess_df = pd.DataFrame.from_dict(emotion_df)

Tess_df.rename(columns={0: "Emotion", 1: "Path"}, inplace=True)

Tess_df.head()

"""**SAVEE DATASET**"""

Savee = "datasets/Savee"

savee_directory_list = os.listdir(Savee)

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

#df = pd.concat([Savee_df], axis=0)
df = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)

df.head(10)

df.drop(df.index[df['Emotion'] == 'surprise'], inplace=True) #dropping it as it had less audio samples compared to other emotions

"""DATA AUGMENTATION"""

def noise(data, random=False, rate=0.035, threshold=0.075):
    if random:
        rate = np.random.random() * threshold
    noise_amp = rate*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data, rate=1000):
    shift_range = int(np.random.uniform(low=-5, high = 5)*rate)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=2.0, random=False):
    if random:
        pitch_factor = np.random.uniform(-pitch_factor, pitch_factor)  # Randomize within range
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

n_fft = 2048
hop_length = 512

def zcr(data, frame_length=2048, hop_length=512):
    zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def energy(data, frame_length=2048, hop_length=512):
    en = np.array([np.sum(np.power(np.abs(data[hop:hop+frame_length]), 2)) for hop in range(0, data.shape[0], hop_length)])
    return en / frame_length


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def entropy_of_energy(data, frame_length=2048, hop_length=512):
    energies = energy(data, frame_length, hop_length)
    energies /= np.sum(energies)

    entropy = 0.0
    entropy -= energies * np.log2(energies)
    return entropy


def spc(data, sr, frame_length=2048, hop_length=512):
    spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spectral_centroid)

def spc_entropy(data, sr):
    spc_en = entropy.spectral_entropy(data, sf=sr, method="fft")
    return spc_en

def spc_flux(data):
    isSpectrum = data.ndim == 1
    if isSpectrum:
        data = np.expand_dims(data, axis=1)

    X = np.c_[data[:, 0], data]
    af_Delta_X = np.diff(X, 1, axis=1)
    vsf = np.sqrt((np.power(af_Delta_X, 2).sum(axis=0))) / X.shape[0]

    return np.squeeze(vsf) if isSpectrum else vsf


def spc_rollof(data, sr, frame_length=2048, hop_length=512):
    spcrollof = librosa.feature.spectral_rolloff(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(spcrollof)


def chroma_stft(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sr)
    return np.squeeze(chroma_stft.T) if not flatten else np.ravel(chroma_stft.T)


def mel_spc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    return np.squeeze(mel.T) if not flatten else np.ravel(mel.T)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_feature.T) if not flatten else np.ravel(mfcc_feature.T)

def extract_features(data, sr, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((
        result,
        zcr(data, frame_length, hop_length),
        np.mean(energy(data, frame_length, hop_length), axis=0),
        np.mean(entropy_of_energy(data, frame_length, hop_length), axis=0),
        rmse(data, frame_length, hop_length),
        spc(data, sr, frame_length, hop_length),
        spc_flux(data),
        spc_rollof(data, sr, frame_length, hop_length),
        chroma_stft(data, sr, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))

    spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
    result = np.hstack((result, np.mean(spectral_contrast, axis=1)))

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr)
    result = np.hstack((result, np.mean(tonnetz, axis=1)))

    mfcc_delta = librosa.feature.delta(librosa.feature.mfcc(y=data, sr=sr))
    result = np.hstack((result, np.mean(mfcc_delta, axis=1)))

    return result

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

# currently used in cnn lstm / 7 layer and 8 layer cnn models

def get_features_new(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset)
    file_name = os.path.basename(path)

    features = extract_features(data, sample_rate)
    result = np.array(features)

    # Data augmentation: Noise
    noise_data = noise(data, random=True)
    features_noise = extract_features(noise_data, sample_rate)
    result = np.vstack((result, features_noise))

    # Data augmentation: Pitch shifting
    pitched_data = pitch(data, sample_rate, random=True)
    features_pitch = extract_features(pitched_data, sample_rate)
    result = np.vstack((result, features_pitch))

    # Data augmentation: Pitch shifting + Noise
    data_noise_pitch = noise(pitch(data, sample_rate, random=True), random=True)
    features_noise_pitch = extract_features(data_noise_pitch, sample_rate)
    result = np.vstack((result, features_noise_pitch))

    return result

# currently used in cnn lstm which outputs predicted csv with filenames.

def get_features_new_final(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset)
    file_name = os.path.basename(path)

    # Extract features (manual + new features)
    features = extract_features(data, sample_rate)


    return features, file_name

X, Y, filenames = [], [], []
print("Processing manual features...")

for ind, (path, emotion) in enumerate(zip(df.Path, df.Emotion)):
    features, file_name = get_features_new_final(path)
    X.append(features)
    Y.append(emotion)
    filenames.append(file_name)
    if ind % 100 == 0:
        print(f"{ind} samples have been processed...")

feature_columns = [f"Feature_{i+1}" for i in range(len(X[0]))]
extracted_df = pd.DataFrame(X, columns=feature_columns)

extracted_df["labels"] = Y
extracted_df["File_Name"] = filenames
#debugging
print("Previewing DataFrame before saving:")
print(extracted_df.head())
print("Columns in the DataFrame:", extracted_df.columns)

features_path = "feature_csv/features_cnn.csv"
extracted_df.to_csv(features_path, index=False)

print("Features and labels saved to:", features_path)
extracted_df.head()

# Loading features
extracted_df = pd.read_csv(features_path)
print(extracted_df.shape)
print(extracted_df['labels'].unique())

filenames = extracted_df["File_Name"].values

# Handling missing values
extracted_df = extracted_df.fillna(0)
print(extracted_df.isna().any())
print(extracted_df.shape)
extracted_df.head()

# Separating features, labels
X = extracted_df.drop(labels=["labels", "File_Name"], axis=1)
Y = extracted_df["labels"]

# Encoding labels
lb = LabelEncoder()
Y_encoded = lb.fit_transform(Y)
Y_categorical = to_categorical(Y_encoded)
print(lb.classes_)
print(Y_categorical.shape)

# Splitting dataset into train and test sets, including filenames
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X, Y_categorical, filenames, random_state=42, test_size=0.2, shuffle=True
)

X_train, X_val, y_train, y_val, filenames_train, filenames_val = train_test_split(
    X_train, y_train, filenames_train, random_state=42, test_size=0.1, shuffle=True
)

print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)


# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
print(X_train_scaled.shape, X_test_scaled.shape, X_val_scaled.shape)

# Reshaping for CNN/LSTM input
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_val_scaled = np.expand_dims(X_val_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)
print(X_train_scaled.shape)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Defining cnn-lstm hybrid model
def hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # cnn layers
    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool1")(x)


    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool2")(x)

    x = layers.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="batch_norm3")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool3")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="batch_norm4")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding='same', name="max_pool4")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="batch_norm5")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool5")(x)

    # lstm Layer
    x = layers.LSTM(128, return_sequences=False, name="lstm")(x)

    x = layers.Dense(512, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="batch_norm6")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Model")

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", f1_m])

    return model

input_shape = (X_train_scaled.shape[1], 1)
num_classes = 6

model = hybrid_model(input_shape=input_shape, num_classes=num_classes)

model.summary()


earlystopping = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Trainning the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[earlystopping, learning_rate_reduction]
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Training- validation loss plot
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if train_loss and val_loss:
    ax[0].plot(train_loss, label='Training Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

# Training- validation accuracy plot
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
if train_acc and val_acc:
    ax[1].plot(train_acc, label='Training Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

fig.tight_layout()
plt.savefig('loss_acc_cnn_lstm.png')
plt.show()

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")

# gave less accuracy as data augmentation not considered with it when getting features extracted for this

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

filenames_test = extracted_df['File_Name'].iloc[X_test.index].values

results_df = pd.DataFrame({
    'File_Name': filenames_test,
    'True_Label': y_true,
    'Predicted_Label': y_pred_classes
})

results_df['True_Label'] = lb.inverse_transform(results_df['True_Label'])
results_df['Predicted_Label'] = lb.inverse_transform(results_df['Predicted_Label'])

results_path = 'predicted_emotions/test_predictions_cnn_lstm.csv'
results_df.to_csv(results_path, index=False)

print(f"Predictions saved to {results_path}")



cm = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# this is the above generated confusion matrix, manually converting it to individual accuracy bar plot.
cm = np.array([
    [256, 22, 13, 43,  9,  3],
    [ 27, 178, 22, 37, 29, 29],
    [ 16,  24, 192, 54, 26, 38],
    [ 34,  27,  31, 217, 28,  8],
    [  6,  44,  17, 33, 219, 26],
    [  6,  50,  33, 15, 81, 184],
])

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

individual_accuracies = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(10, 6))
plt.bar(cm_plot_labels, individual_accuracies, color='purple', alpha=0.7)
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Individual Accuracies for Each Emotion', fontsize=15)
plt.ylim(0, 1)
for i, acc in enumerate(individual_accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/individual_accuracies_cnn_lstm.png')
plt.show()

"""**from here also considering data augmentaion and using get_features_new method written above for feature extraction, this is computationaly expensive but gives good accuracy. HAVING 4 SETS OF FEATURES FOR EACH DATASET**"""

X, Y = [], []
print("Feature processing...")
for path, emotion, ind in zip(df.Path, df.Emotion, range(df.Path.shape[0])):
    features = get_features_new(path)
    for ele in features:
        X.append(ele)
        Y.append(emotion)
    if ind%100 == 0:
            print(f"{ind} samples has been processed...")
#     if ind == 6:
#         break
print("Done.")

features_path = "feature_csv/feature_cnn_final1.csv"

extracted_df = pd.DataFrame(X)
extracted_df["labels"] = Y
extracted_df.to_csv(features_path, index=False)
extracted_df.head()

# Loading features
extracted_df = pd.read_csv(features_path)
print(extracted_df.shape)
print(extracted_df['labels'].unique())

# Handling missing values
extracted_df = extracted_df.fillna(0)
print(extracted_df.isna().any())
print(extracted_df.shape)
extracted_df.head()

# Separating features, labels
X = extracted_df.drop(labels=["labels"], axis=1)
Y = extracted_df["labels"]

# Encoding labels
lb = LabelEncoder()
Y_encoded = lb.fit_transform(Y)
Y_categorical = to_categorical(Y_encoded)
print(lb.classes_)
print(Y_categorical.shape)

# Split dataset into train and test sets, including filenames
X_train, X_test, y_train, y_test = train_test_split(
    X, Y_categorical, random_state=42, test_size=0.2, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,  random_state=42, test_size=0.1, shuffle=True
)

print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)


# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
print(X_train_scaled.shape, X_test_scaled.shape, X_val_scaled.shape)

# Reshaping for CNN/LSTM input
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_val_scaled = np.expand_dims(X_val_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)
print(X_train_scaled.shape)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# lstm cnn model with data augmented features as input

# Defining cnn lstm model
def hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # cnn layers
    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool1")(x)

    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool2")(x)

    x = layers.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="batch_norm3")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool3")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="batch_norm4")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding='same', name="max_pool4")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="batch_norm5")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool5")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv6")(x)
    x = layers.BatchNormalization(name="batch_norm6")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool6")(x)

    # lstm layer
    x = layers.LSTM(128, return_sequences=False, name="lstm1")(x)

    # fully connected layer
    x = layers.Dense(512, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="batch_norm7")(x)
    x = layers.Dropout(0.4, name="dropout1")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Model")

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", f1_m])

    return model

input_shape = (X_train_scaled.shape[1], 1)
num_classes = 6

model = hybrid_model(input_shape=input_shape, num_classes=num_classes)

model.summary()

earlystopping = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Trainning above model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[earlystopping, learning_rate_reduction]
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Training- validation loss plot
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if train_loss and val_loss:
    ax[0].plot(train_loss, label='Training Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

# Training-validation accuracy plot
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
if train_acc and val_acc:
    ax[1].plot(train_acc, label='Training Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

fig.tight_layout()
plt.savefig('plots/loss_acc_cnn_lstm.png')
plt.show()

path_to_model = "models/resnewlstmcnn_model.h5"
model.save(path_to_model)
print(f"Model saved at {path_to_model}")

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")
# this time accuracy increased and could further increase if trained for higher epoochs. as used data augmentation with feature extraction

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/confusion_matrix_cnn_lstm.png')


cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

"""**7 cnn layer model to increase accuracy further, as lstm couldn't increase accuracy too high**"""

# Defining cnn 7 layer model
def hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # cnn layers
    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool1")(x)


    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool2")(x)

    x = layers.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="batch_norm3")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool3")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="batch_norm4")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding='same', name="max_pool4")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="batch_norm5")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool5")(x)


    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv6")(x)
    x = layers.BatchNormalization(name="batch_norm6")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool6")(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation="relu", name="conv7")(x)
    x = layers.BatchNormalization(name="batch_norm7")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool7")(x)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(512, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="batch_norm8")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Model")

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", f1_m])

    return model

input_shape = (X_train_scaled.shape[1], 1)
num_classes = 6

model = hybrid_model(input_shape=input_shape, num_classes=num_classes)

model.summary()

earlystopping = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Trainning 7 layer model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[earlystopping, learning_rate_reduction]
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Training- validation loss plot
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if train_loss and val_loss:
    ax[0].plot(train_loss, label='Training Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

# Training - validation accuracy plot
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
if train_acc and val_acc:
    ax[1].plot(train_acc, label='Training Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

fig.tight_layout()
plt.savefig('plots/loss_acc_cnn7layer.png')
plt.show()

path_to_model = "models/cnn7layer_model.h5"
model.save(path_to_model)
print(f"Model saved at {path_to_model}")

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")
# accuracy increased highly - 94.25%

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

#plotting cm
cm = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/confusion_matrix_cnn7layer.png')

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# manually converting the above cm into bar plot with individual accuracies
cm = np.array([
    [1278,  11,  18,  37,   8,   9],
    [ 24, 1280,  19,  15,  17,  17],
    [ 14,  12, 1291,  24,   7,  24],
    [ 31,  22,  31, 1266,  26,   7],
    [  5,  10,  18,   7, 1373,  22],
    [  2,  12,  23,  11,  20, 1317]
])

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

individual_accuracies = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(10, 6))
plt.bar(cm_plot_labels, individual_accuracies, color='purple', alpha=0.7)
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Individual Accuracies for Each Emotion', fontsize=15)
plt.ylim(0, 1)

for i, acc in enumerate(individual_accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/individual_accuracies_cnn7layer.png')
plt.show()

"""**7 layer cnn and using l2 regularization and dropout layer to further increase accuracy**"""


def hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # cnn layers with L2 regularization
    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv1")(inputs)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool1")(x)

    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv2")(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool2")(x)

    x = layers.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv3")(x)
    x = layers.BatchNormalization(name="batch_norm3")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool3")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv4")(x)
    x = layers.BatchNormalization(name="batch_norm4")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool4")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv5")(x)
    x = layers.BatchNormalization(name="batch_norm5")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same", name="max_pool5")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv6")(x)
    x = layers.BatchNormalization(name="batch_norm6")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same", name="max_pool6")(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu", kernel_regularizer=l2(0.01), name="conv7")(x)
    x = layers.BatchNormalization(name="batch_norm7")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same", name="max_pool7")(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation="relu", kernel_regularizer=l2(0.01), name="conv8")(x)
    x = layers.BatchNormalization(name="batch_norm8")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool8")(x)

    # glob avg pooling
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.01), name="dense3")(x)
    x = layers.BatchNormalization(name="batch_norm11")(x)
    x = layers.Dropout(0.5)(x)  # dropout layer to reduce any overfitting

    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_Model_with_l2_regularization")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1_m])

    return model

input_shape = (X_train_scaled.shape[1], 1)
num_classes = 6

model = hybrid_model(input_shape=input_shape, num_classes=num_classes)

model.summary()

earlystopping = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Trainning the above model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[earlystopping, learning_rate_reduction]
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Training-validation loss
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if train_loss and val_loss:
    ax[0].plot(train_loss, label='Training Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

# Training-validation accuracy
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
if train_acc and val_acc:
    ax[1].plot(train_acc, label='Training Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

fig.tight_layout()
plt.savefig('plots/loss_acc_cnn_with_reg_dropout.png')
plt.show()

path_to_model = "models/cnnadvlayer_model.h5"
model.save(path_to_model)
print(f"Model saved at {path_to_model}")

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")
# l2 regularizarion and dropout layer decreased the accuracy - 67%

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

#plotting cm
cm = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/confusion_matrix_cnn7layer_with_reg_dropout.png')

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

"""**cnn 8 layer to increase accuracy given by 7 layer further**"""

# Defining 8 layer model
def hybrid_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # cnn layer
    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="batch_norm1")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool1")(x)


    x = layers.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="batch_norm2")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool2")(x)

    x = layers.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="batch_norm3")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding="same", name="max_pool3")(x)

    x = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="batch_norm4")(x)
    x = layers.MaxPooling1D(pool_size=5, strides=2, padding='same', name="max_pool4")(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="batch_norm5")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool5")(x)


    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation="relu", name="conv6")(x)
    x = layers.BatchNormalization(name="batch_norm6")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool6")(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation="relu", name="conv7")(x)
    x = layers.BatchNormalization(name="batch_norm7")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool7")(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation="relu", name="conv8")(x)
    x = layers.BatchNormalization(name="batch_norm8")(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name="max_pool8")(x)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(512, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="batch_norm9")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Model")

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", f1_m])

    return model

input_shape = (X_train_scaled.shape[1], 1)
num_classes = 6

model = hybrid_model(input_shape=input_shape, num_classes=num_classes)

model.summary()

earlystopping = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Trainning above 8 layer model to 60 epochs
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=60,
    batch_size=32,
    callbacks=[earlystopping, learning_rate_reduction]
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Training- validation loss
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if train_loss and val_loss:
    ax[0].plot(train_loss, label='Training Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

# Training-validation accuracy
train_acc = history.history.get('accuracy', [])
val_acc = history.history.get('val_accuracy', [])
if train_acc and val_acc:
    ax[1].plot(train_acc, label='Training Accuracy')
    ax[1].plot(val_acc, label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")

fig.tight_layout()
plt.savefig('plots/loss_acc_cnn8layer.png')
plt.show()

path_to_model = "models/cnn8layer_model.h5"
model.save(path_to_model)
print(f"Model saved at {path_to_model}")

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")
# this 8 layer almost gave same accuracy as 7 layer even when trained for 60 epochs - 93%

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# plotting cm
cm = confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('plots/confusion_matrix_cnn_8layer.png')

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# manually coverting this above generated cm into bar plot with individual accuracy for emotions
cm = np.array([
    [1285,  26,  21,  20,   4,   5],
    [ 22, 1248,  26,  31,  17,  28],
    [ 15,  19, 1252,  35,  18,  33],
    [ 41,  26,  48, 1225,  28,  15],
    [  3,  20,  22,  15, 1340,  35],
    [  5,  23,  29,  15,  38, 1275]
])

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

individual_accuracies = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(10, 6))
plt.bar(cm_plot_labels, individual_accuracies, color='purple', alpha=0.7)
plt.xlabel('Emotions', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Individual Accuracies for Each Emotion', fontsize=15)
plt.ylim(0, 1)

for i, acc in enumerate(individual_accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/individual_accuracies_cnn_8layer.png')
plt.show()