import os
import logging
import asyncio
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, ResultReason
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from dotenv import load_dotenv
import openai
import sqlite3

# Custom imports
import audio_classification
import audio_to_spectrogram
import speech_emotion_recognition

# 加載環境變量
load_dotenv()

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 Flask 應用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 設置Azure語音配置
def setup_azure_speech(azure_speech_key, azure_service_region):
    speech_config = SpeechConfig(subscription=azure_speech_key, region=azure_service_region)
    audio_config = AudioOutputConfig(use_default_speaker=True)
    return SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# 初始化配置
def setup_configuration():
    openai.api_key = os.getenv('OPENAI_API_KEY')
    azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
    azure_service_region = os.getenv('AZURE_SERVICE_REGION')

    if not openai.api_key or not azure_speech_key or not azure_service_region:
        raise ValueError("需要設置環境變量: OPENAI_API_KEY, AZURE_SPEECH_KEY, AZURE_SERVICE_REGION")
    
    return azure_speech_key, azure_service_region

# 初始化數據庫
def initialize_database():
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS user_data (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     username TEXT,
                     emotion TEXT,
                     log_file TEXT)''')
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"數據庫錯誤: {e}")
    finally:
        conn.close()

# 檢查文件類型
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 生成GPT問候語
def generate_greeting_with_gpt(name, emotion):
    prompt = f"根據以下使用者的姓名和情感生成一個適合的問候語：\n\n姓名: {name}\n情感: {emotion}\n\n"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "你擅長關心，會先向使用者問好並關心使用者目前的情緒狀態。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )
    
    return response['choices'][0]['message']['content'].strip()

# 異步即時串流TTS
async def stream_tts(synthesizer, text):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, synthesizer.speak_text_async, text)
        result = result.get()
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            logging.info("TTS合成成功並即時播放")
        else:
            logging.error(f"TTS合成失敗: {result.reason}")
    except Exception as e:
        logging.error(f"TTS即時串流失敗: {e}")

# 保存數據到數據庫
def save_to_database(username, emotion, log_file):
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        c.execute("INSERT INTO user_data (username, emotion, log_file) VALUES (?, ?, ?)",
                  (username, emotion, log_content))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"數據庫錯誤: {e}")
    finally:
        conn.close()
        os.remove(log_file)

# 從數據庫加載數據
def load_from_database(username):
    try:
        conn = sqlite3.connect('user_data.db')
        c = conn.cursor()
        c.execute("SELECT emotion, log_file FROM user_data WHERE username = ? ORDER BY id DESC LIMIT 1", (username,))
        data = c.fetchone()
        return data
    except sqlite3.Error as e:
        logging.error(f"數據庫錯誤: {e}")
        return None
    finally:
        conn.close()

# GPT對話循環
async def gpt_chat_loop(initial_prompt, username, emotion, azure_speech_key, azure_service_region):
    messages = [
        {"role": "system", "content": "你是一個專業的語文教學，會根據使用者的情緒和想要學習的語言先嘗試了解使用者的程度並給予合適的教材，你的回覆是交給Azure的TexttoSpeech播放語音所以請生成合適的回覆內容並取消標點符號。"},
        {"role": "user", "content": initial_prompt}
    ]

    log_file = f"{username}_chat_log.txt"
    previous_data = load_from_database(username)
    if (previous_data is not None) and (previous_data[1] is not None):
        previous_emotion, previous_log = previous_data
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(previous_log)
        
        messages.append({"role": "system", "content": f"過去的情感狀態是 {previous_emotion}。"})
        messages.append({"role": "system", "content": "以下是上次的對話記錄。"})

        with open(log_file, 'r', encoding='utf-8') as f:
            previous_conversation = f.read()
            messages.append({"role": "system", "content": previous_conversation})
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"GPT: {initial_prompt}\n")

    synthesizer = setup_azure_speech(azure_speech_key, azure_service_region)

    while True:
        user_input = speech_to_text()
        if user_input:
            if "退出" in user_input:
                logging.info("對話結束")
                save_to_database(username, emotion, log_file)
                break

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"User: {user_input}\n")
            
            messages.append({"role": "user", "content": user_input})
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1500
            )
            gpt_response = response['choices'][0]['message']['content'].strip()
            logging.info(f"GPT: {gpt_response}")

            # 將GPT回覆寫入聊天日誌並播放
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"GPT: {gpt_response}\n")

            await stream_tts(synthesizer, gpt_response)

            messages.append({"role": "assistant", "content": gpt_response})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            asyncio.run(process_audio(file_path))
            return jsonify(message='File processed successfully'), 200
        except Exception as e:
            logging.error(f"處理音頻時發生錯誤: {e}")
            return jsonify(error=str(e)), 500

async def process_audio(file_path):
    azure_speech_key, azure_service_region = setup_configuration()

    audio_probs = audio_classification.classify_audio(file_path)
    audio_weight = 0.5

    spectrogram_probs = audio_to_spectrogram.classify_spectrogram(file_path)
    spectrogram_weight = 0.5

    final_probs = (audio_probs * audio_weight) + (spectrogram_probs * spectrogram_weight)
    final_label = np.argmax(final_probs, axis=1)[0]
    final_confidence = final_probs[0][final_label]
    username = audio_classification.class_names[final_label]

    emotion_result = speech_emotion_recognition.recognize_emotion(file_path)
    emotion = emotion_result.strip()

    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Username - {username}\n")
        f.write(f"Final classification - {audio_classification.class_names[final_label]}, Confidence: {final_confidence:.2f}\n")
        f.write(f"Emotion - {emotion}\n")

    with open('results.txt', 'r', encoding='utf-8') as f:
        results_content = f.read().splitlines()

    username = results_content[0].split(" - ")[1]
    emotion_detail = results_content[2].split(" - ")[1]

    greeting = generate_greeting_with_gpt(username, emotion_detail)
    synthesizer = setup_azure_speech(azure_speech_key, azure_service_region)
    await stream_tts(synthesizer, greeting)

    await gpt_chat_loop(greeting, username, emotion_detail, azure_speech_key, azure_service_region)

if __name__ == "__main__":
    initialize_database()
    app.run(host='0.0.0.0', port=5000)
