# Emotion to Music Matching & Generation
# Please note that you need to comment in line 12 (os.enviorn[...]) if running on Google Colab

from IPython.display import Audio
import time
import pandas as pd
import os
import pygame
import matplotlib.pyplot as plt
import seaborn as sns

# os.environ['SDL_AUDIODRIVER'] = 'dsp' # ONLY FOR COLAB - COMMENT OUT FOR LOCAL RUN

emotion_valence_arousal = {
    "neutral": {"valence": (4.5, 6.2), "arousal": (4.8, 5.8)},
    "happy": {"valence": (6.3, 10.0), "arousal": (5.8, 10.0)},
    "sad": {"valence": (1.0, 4.5), "arousal": (1.0, 5.5)},
    "angry": {"valence": (1.0, 3.0), "arousal": (7.0, 10.0)},
    "fear": {"valence": (4.8, 5.2), "arousal": (3.8, 4.2)}
}

emotion_va_df = pd.DataFrame([
    {
        'emotion': emotion,
        'valence_min': ranges['valence'][0],
        'valence_max': ranges['valence'][1],
        'arousal_min': ranges['arousal'][0],
        'arousal_max': ranges['arousal'][1]
    }
    for emotion, ranges in emotion_valence_arousal.items()
])

# Loading the Music Database
deam_va_values_csv = "datasets/DEAM/static_annotations.csv"
deam_va_values = pd.read_csv(deam_va_values_csv)
audio_path = "datasets/DEAM/MEMD_audio"
valid_song_ids = set(deam_va_values['song_id'])

audio_files = []
for file in os.listdir(audio_path):
    if file.endswith(".mp3"):
        song_id = int(file.split(".")[0])
        if song_id in valid_song_ids:
            audio_files.append({'song_id': song_id, 'file_path': os.path.join(audio_path, file)})

audio_df = pd.DataFrame(audio_files)
audio_with_va_df = pd.merge(audio_df, deam_va_values[['song_id', "mean_arousal", 'mean_valence']], on='song_id', how='inner')
audio_with_va_df = audio_with_va_df.sort_values('song_id')
print(audio_with_va_df.head())

# Save the mapping
audio_with_va_df.to_csv("datasets/DEAM/outputs/deam_audio_mapping.csv", index=False)
audio_with_emotions_df = audio_with_va_df.copy()

# Map the Emotion to the Valence/Arousal Space
def map_emotion_to_va(valence, arousal):
    for emotion, ranges in emotion_valence_arousal.items():
        val_range = ranges['valence']
        aro_range = ranges['arousal']
        if (val_range[0] <= valence <= val_range[1]) and (aro_range[0] <= arousal <= aro_range[1]):
            return emotion

    # Using Euclidean Distance to find the closest emotion if there is no match
    min_dist = float('inf')
    closest_emotion = None
    for emotion, ranges in emotion_valence_arousal.items():
        if emotion not in ['fear']:         # We are not using fear to be closest as it is a very specific emotions and MUST be in the range given
            val_mid = (ranges['valence'][0] + ranges['valence'][1]) / 2
            aro_mid = (ranges['arousal'][0] + ranges['arousal'][1]) / 2

            dist = ((valence - val_mid) ** 2 + (arousal - aro_mid) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_emotion = emotion

    return closest_emotion

audio_with_emotions_df['emotion'] = audio_with_emotions_df.apply(lambda row: map_emotion_to_va(row['mean_valence'], row['mean_arousal']), axis=1)

# Plot the Valence vs. Arousal Distribution by Emotion
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=audio_with_emotions_df,
    x="mean_valence",
    y="mean_arousal",
    hue="emotion",
    palette="viridis",
    alpha=0.7
)
plt.title("Valence vs. Arousal Distribution by Emotion", fontsize=14)
plt.xlabel("Mean Valence", fontsize=12)
plt.ylabel("Mean Arousal", fontsize=12)
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("plots/valence_arousal_emotion.png")

audio_with_emotions_df.to_csv("datasets/DEAM/outputs/deam_mapping_with_emotions.csv", index=False)
pred_df = pd.read_csv("predicted_emotions/predicted_emotions.csv")

# Obtains a music file for the emotion given
def music_to_emotions (df, pred_df, emotion_list=None):
    if emotion_list is None:
        emotion_list = list(emotion_valence_arousal.keys())
    elif isinstance(emotion_list, str):
        emotion_list = [emotion_list]

    if os.environ.get('COLAB_RELEASE_TAG'): # When ran in Colab
        for emotion in emotion_list:
            matching_songs = df[df['emotion'] == emotion]
            song = matching_songs.sample(n=1).iloc[0]

            matching_preds = pred_df[pred_df['True_Emotion'] == emotion]
            if not matching_preds.empty:
                pred = matching_preds.sample(n=1).iloc[0]
                print(f"Input File Name: {pred['File_Name']}")
                print(f"Predicted Emotion of ID: {pred['Predicted_Emotion']}")
                print(f"True Emotion: {pred['True_Emotion']} \n")

            print(f"Playing song ID: {song['song_id']}")
            print(f"Song Emotion: {song['emotion']}")
            print(f"Valence: {song['mean_valence']}")
            print(f"Arousal: {song['mean_arousal']}")


            try:
                # For Colab, just use IPython.display.Audio
                audio_player = Audio(song['file_path'])
                display(audio_player)
                print("---------")
            except Exception as e:
                print(f"Error: {e}")

    else: # When ran locally
        for emotion in emotion_list:
            matching_songs = df[df['emotion'] == emotion]
            song = matching_songs.sample(n=1).iloc[0]

            matching_preds = pred_df[pred_df['True_Emotion'] == emotion]
            if not matching_preds.empty:
                pred = matching_preds.sample(n=1).iloc[0]
                print(f"Input File Name: {pred['File_Name']}")
                print(f"Predicted Emotion of ID: {pred['Predicted_Emotion']}")
                print(f"True Emotion: {pred['True_Emotion']} \n")

            print(f"Playing song ID: {song['song_id']}")
            print(f"Song Emotion: {song['emotion']}")
            print(f"Valence: {song['mean_valence']}")
            print(f"Arousal: {song['mean_arousal']}")
            print("---------")

            try:
                # Only runs pygame if not in Colab
                pygame.init()
                pygame.display.set_mode((100, 100))
                pygame.mixer.init()
                pygame.mixer.music.load(song['file_path'])
                pygame.mixer.music.play()

                print("Press 'X' to stop the music.")
                running = True
                while running:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                            pygame.mixer.music.stop()
                            running = False
                    if not pygame.mixer.music.get_busy():
                        running = False
                    time.sleep(0.5)

                pygame.mixer.quit()
                pygame.display.quit()
                pygame.quit()
            except Exception as e:
                print(f"Error: {e}")
                pygame.mixer.quit()
                pygame.display.quit()
                pygame.quit()


# Please run the sections below to generate music
music_to_emotions(audio_with_emotions_df, pred_df)

# # OR run the sections below to generate music for specific emotions
# music_to_emotions(audio_with_emotions_df, pred_df, ['happy'])
# music_to_emotions(audio_with_emotions_df, pred_df, ['sad'])
# music_to_emotions(audio_with_emotions_df, pred_df, ['neutral'])
# music_to_emotions(audio_with_emotions_df, pred_df, ['angry'])
# music_to_emotions(audio_with_emotions_df, pred_df, ['fear'])

