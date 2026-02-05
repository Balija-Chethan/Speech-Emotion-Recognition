# Speech Emotion Recognition

This project detects human emotions from speech audio using machine learning.

## What this project does
- Takes speech audio files as input
- Extracts pitch-based features
- Trains a machine learning model
- Predicts emotions like Angry, Fear, Neutral, etc.

## Files in this project
- symbolic_pitch_emo_db.py : extracts features from audio
- add_labels.py : adds emotion labels
- train_model.py : trains and tests the ML model
- features/ : contains CSV files and confusion matrix image

## Model used
- Random Forest Classifier

## Result
- Accuracy achieved: ~59%
- Confusion matrix is generated for evaluation

## Note
Dataset is not uploaded due to GitHub size limits.
