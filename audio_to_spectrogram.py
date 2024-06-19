# audio_to_spectrogram.py
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import noisereduce as nr
import os

class_names = ['白東衢', '林靄君', '李宇芯', '蔡承哲']

def denoise_audio(y, sr):
    noise_sample = y[:int(0.5 * sr)]
    return nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)

def amplify_audio(y, gain_dB):
    gain = 10 ** (gain_dB / 20)
    return y * gain

def normalize_audio(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y

def plot_and_save_spectrogram(y, sr, start_time, duration, output_file):
    start_sample = int(start_time * sr)
    end_sample = start_sample + int(duration * sr)
    y_segment = y[start_sample:end_sample]
    
    n_fft = 1024
    hop_length = 256
    D = librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(3, 3), dpi=200)
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
    plt.axis('off')
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def mp3_to_spectrogram(file_path, output_dir, gain_dB, segment_length=0.1):
    y, sr = librosa.load(file_path)
    y = denoise_audio(y, sr)
    y = amplify_audio(y, gain_dB)
    y = normalize_audio(y)
    
    segment_samples = int(segment_length * sr)
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]
        if len(segment) < segment_samples:
            break
        
        output_file = os.path.join(output_dir, f"spectrogram_{i}.png")
        plot_and_save_spectrogram(segment, sr, 0, segment_length, output_file)

def preprocess_image(image_path, target_size=(150, 150)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def load_picture_model(model_path='best_picture.keras'):
    return load_model(model_path)

def classify_spectrograms(file_path, model_path='best_picture.keras', output_dir='output_spectrograms'):
    model = load_picture_model(model_path)
    mp3_to_spectrogram(file_path, output_dir, gain_dB=10)
    
    predicted_probs = None
    num_segments = 0
    i = 0
    
    while True:
        image_path = os.path.join(output_dir, f"spectrogram_{i}.png")
        if not os.path.exists(image_path):
            break
        
        image_data = preprocess_image(image_path)
        segment_probs = model.predict(image_data)
        
        if predicted_probs is None:
            predicted_probs = segment_probs
        else:
            predicted_probs += segment_probs
        num_segments += 1
        i += 1
    
    if num_segments > 0:
        average_probs = predicted_probs / num_segments
        return average_probs
    else:
        return np.zeros((1, len(class_names)))

def classify_spectrogram(file_path, model_path='best_picture.keras'):
    predicted_probs = classify_spectrograms(file_path, model_path)
    return predicted_probs
