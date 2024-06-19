# audio_classification.py
from tensorflow.keras.models import load_model
import numpy as np
import librosa

# 定義原始類別名稱對照
class_names = ['白東衢', '林靄君', '李宇芯', '蔡承哲']

def load_audio_model(model_path='best_audio.keras'):
    return load_model(model_path)

def extract_features(y_data, sr, n_mfcc=13, fixed_length=160):
    mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=y_data, sr=sr)
    mel = librosa.feature.melspectrogram(y=y_data, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y_data, sr=sr)
    features = np.hstack((
        np.mean(mfcc, axis=1), 
        np.mean(chroma, axis=1), 
        np.mean(mel, axis=1), 
        np.mean(contrast, axis=1)
    )).flatten()
    
    if len(features) < fixed_length:
        features = np.pad(features, (0, fixed_length - len(features)), mode='constant')
    else:
        features = features[:fixed_length]
    
    return features

def test_new_audio(model, file_path):
    y_data, sr = librosa.load(file_path)
    features = extract_features(y_data, sr)
    features = np.expand_dims(features, axis=0)
    
    predicted_probs = model.predict(features)
    return predicted_probs

def classify_audio(file_path, model_path='best_audio.keras'):
    model = load_audio_model(model_path)
    predicted_probs = test_new_audio(model, file_path)
    return predicted_probs
