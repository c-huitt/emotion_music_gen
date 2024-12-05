import os
import re
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from keras import layers, models, utils, backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model




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

"""**RAVDESS DATASET**"""


Ravdess = "datasets/Ravdess"
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

Ravdess_df = pd.DataFrame.from_dict(emotion_df)
Ravdess_df.rename(columns={1 : "Path", 0 : "Emotion"}, inplace=True)

Ravdess_df.Emotion.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()

"""**CREMA DATASET**"""

Crema = "datasets/Crema"
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

"""**TESS DATASET**"""

Tess = "datasets/Tess"
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

df = pd.concat([ Savee_df, Ravdess_df,  Crema_df, Tess_df], axis=0)
df.head(10)
df.drop(df.index[df['Emotion'] == 'surprise'], inplace=True)

"""**DATA AUGMENTATION**"""

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

"""**MANUAL AUDIO FEATURES**"""

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

# GET FEATURES TO GET THE MANUAL AUDIO FEATURES

def get_features_new_final(path, duration=2.5, offset=0.6):
    data, sample_rate = librosa.load(path, duration=duration, offset=offset)
    file_name = os.path.basename(path)

    features = extract_features(data, sample_rate)
    result = [features]

    noise_data = noise(data, random=True)
    result.append(extract_features(noise_data, sample_rate))

    pitched_data = pitch(data, sample_rate, random=True)
    result.append(extract_features(pitched_data, sample_rate))

    data_noise_pitch = noise(pitch(data, sample_rate, random=True), random=True)
    result.append(extract_features(data_noise_pitch, sample_rate))

    return np.vstack(result), file_name

#FEATURE EXTRACTION
X, Y, filenames = [], [], []

print("Processing features...")
for ind, (path, emotion) in enumerate(zip(df.Path, df.Emotion)):
    features, file_name = get_features_new_final(path)
    for ele in features:
        X.append(ele)
        Y.append(emotion)
        filenames.append(file_name)
    if ind % 100 == 0:
        print(f"{ind} samples processed...")
print("Feature extraction completed.")

# Saving features and filenames to CSV
features_path = "feature_csv/7layer_cnn_features.csv"
extracted_df = pd.DataFrame(X)
extracted_df["labels"] = Y
extracted_df["File_Name"] = filenames
extracted_df.to_csv(features_path, index=False)
print(f"Features saved to {features_path}")

# Loading features
extracted_df = pd.read_csv(features_path)
print(extracted_df.shape)
print(extracted_df['labels'].unique())

# Handling missing values
extracted_df = extracted_df.fillna(0)

# Separating features, labels
X = extracted_df.drop(labels=["labels", "File_Name"], axis=1)
Y = extracted_df["labels"]
filenames = extracted_df["File_Name"].values

# Encoding labels
lb = LabelEncoder()
Y_encoded = lb.fit_transform(Y)
Y_categorical = to_categorical(Y_encoded)

# Spliting dataset into train, validation, and test sets
X_train, X_test, y_train, y_test, filenames_train, filenames_test = train_test_split(
    X, Y_categorical, filenames, random_state=42, test_size=0.2, shuffle=True
)

X_train, X_val, y_train, y_val, filenames_train, filenames_val = train_test_split(
    X_train, y_train, filenames_train, random_state=42, test_size=0.1, shuffle=True
)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Reshaping for CNN input
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_val_scaled = np.expand_dims(X_val_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

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



# Defining 7 layer cnn model
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

    # flatten layer
    x = layers.Flatten(name="flatten")(x)

    # fully connected layers
    x = layers.Dense(512, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="batch_norm8")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Model")

    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", f1_m])

    return model


# Initializing & training model
input_shape = (X_train_scaled.shape[1], 1)
num_classes = Y_categorical.shape[1]
model = hybrid_model(input_shape, num_classes)
model.summary()

earlystopping = EarlyStopping(monitor="val_accuracy", mode='max', patience=5, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Training model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[earlystopping, learning_rate_reduction]
)

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Creating results DataFrame
results_df = pd.DataFrame({
    "File_Name": filenames_test,
    "True_Label": lb.inverse_transform(y_true_classes),
    "Predicted_Label": lb.inverse_transform(y_pred_classes)
})

results_csv_path = "feature_csv/7layer_cnn_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Predicted results saved to {results_csv_path}")

final_train_accuracy = history.history["accuracy"][-1]
final_val_accuracy = history.history["val_accuracy"][-1]

# Printing final accuracies
print(f"Final Training Accuracy: {final_train_accuracy:.2f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Training and validation loss
train_loss = history.history.get('loss', [])
val_loss = history.history.get('val_loss', [])
if train_loss and val_loss:
    ax[0].plot(train_loss, label='Training Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

# Training and validation accuracy
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

path_to_model = "models/7layer_cnn_model.h5"
model.save(path_to_model)
print(f"Model saved at {path_to_model}")

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
    plt.savefig("plots/confusion_matrix_cnn7layer.png")

cm_plot_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# above confusion matrix, manually converting it to individual accuracy bar graph
cm = np.array([
    [1298 , 14 , 19,  39,  10,  9],
    [ 29, 1274,  20,  16,  17,  26],
    [  10,  11, 1251,  42,  25,  29],
    [ 33,  16,  36, 1322,  27,  8],
    [  7,  12,  13,  19, 1262,  30],
    [ 2,  23,  31,  18,  27, 1283],
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