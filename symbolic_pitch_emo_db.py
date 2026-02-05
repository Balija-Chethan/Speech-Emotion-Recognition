import os
import librosa
import numpy as np
import pandas as pd
from collections import Counter

# -------- PATHS --------
DATA_DIR = r"C:\Users\balij\Desktop\SER_RESEARCH\data\emo_db"
OUTPUT_DIR = r"C:\Users\balij\Desktop\SER_RESEARCH\features"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "symbolic_pitch_features.csv")

# -------- EMOTION MAP --------
emotion_map = {
    "W": "Anger",
    "L": "Boredom",
    "E": "Disgust",
    "A": "Fear",
    "F": "Happiness",
    "T": "Sadness",
    "N": "Neutral"
}

def pitch_to_symbol(f0, mean, std):
    if f0 < mean - std:
        return "VL"
    elif f0 < mean:
        return "L"
    elif f0 < mean + std:
        return "H"
    else:
        return "VH"

results = []

# -------- LOOP OVER FILES --------
for file in os.listdir(DATA_DIR):
    if not file.endswith(".wav"):
        continue

    print("Processing:", file)
    file_path = os.path.join(DATA_DIR, file)

    audio, sr = librosa.load(file_path, sr=None)
    pitch, _, _ = librosa.pyin(audio, fmin=60, fmax=450, sr=sr)
    pitch = pitch[~np.isnan(pitch)]

    if len(pitch) == 0:
        continue

    mean_pitch = np.mean(pitch)
    std_pitch = np.std(pitch)

    symbols = [pitch_to_symbol(f, mean_pitch, std_pitch) for f in pitch]
    transitions = [symbols[i] + "->" + symbols[i+1] for i in range(len(symbols)-1)]
    transition_counts = Counter(transitions)

    emotion_code = file[-5]
    emotion = emotion_map.get(emotion_code, "Unknown")

    row = {
        "file": file,
        "emotion": emotion,
        "mean_pitch": mean_pitch,
        "std_pitch": std_pitch
    }

    for t, c in transition_counts.items():
        row[t] = c

    results.append(row)

# -------- SAVE RESULTS --------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.DataFrame(results)
df.fillna(0, inplace=True)
df.to_csv(OUTPUT_CSV, index=False)

print("\nDONE ✅")
print("File created at:", OUTPUT_CSV)
