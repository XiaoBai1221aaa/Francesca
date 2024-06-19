# speech_emotion_recognition.py
import os
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel

model_name_or_path = "xmj2002/hubert-base-ch-speech-emotion-recognition"
duration = 6
sample_rate = 48000

config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

def id2class(id):
    classes = ["生氣", "害怕", "開心", "中性", "悲傷", "驚訝"]
    return classes[id] if id < len(classes) else "未知"

def weighted_prediction(scores):
    emotions = ["生氣", "害怕", "開心", "中性", "悲傷", "驚訝"]
    weighted_emotions = {emotion: score for emotion, score in zip(emotions, scores)}
    sorted_emotions = sorted(weighted_emotions.items(), key=lambda item: item[1], reverse=True)
    return sorted_emotions

class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

def load_emotion_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
    model = HubertForSpeechClassification.from_pretrained(model_name_or_path, config=config)
    model.eval()
    return processor, model

def predict(path, processor, model):
    speech, sr = librosa.load(path=path, sr=sample_rate)
    
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    speech = (speech - speech.mean()) / (speech.std() + 1e-5)
    
    speech = processor(speech, padding="max_length", truncation=True, max_length=duration * sr,
                       return_tensors="pt", sampling_rate=sr).input_values
    
    with torch.no_grad():
        logits = model(speech)
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    weighted_results = weighted_prediction(scores)
    
    return f"{weighted_results[0][0]} (Confidence: {weighted_results[0][1]:.4f}), "f"可能帶有一點 {weighted_results[1][0]} (Confidence: {weighted_results[1][1]:.4f})\n"

def recognize_emotion(file_path):
    processor, model = load_emotion_model()
    return predict(file_path, processor, model)
