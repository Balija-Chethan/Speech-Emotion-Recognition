import pandas as pd

df = pd.read_csv("features/symbolic_pitch_features.csv")

def get_emotion(filename):
    if "A" in filename:
        return "Angry"
    elif "H" in filename:
        return "Happy"
    elif "S" in filename:
        return "Sad"
    elif "N" in filename:
        return "Neutral"
    elif "W" in filename:
        return "Fear"
    elif "T" in filename:
        return "Disgust"
    else:
        return "Unknown"

df["emotion"] = df["file"].apply(get_emotion)

df.to_csv("features/symbolic_pitch_features_labeled.csv", index=False)

print("DONE ✅ Labeled file created")
