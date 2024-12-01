import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import pickle
import os
import uuid
import pandas as pd
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Load preprocessed embeddings and data
@st.cache(allow_output_mutation=True)
def load_data():
    load_path = r'Saved_state/embeddings.pkl'
    with open(load_path, 'rb') as f:
        question_embeddings, data = pickle.load(f)
    return question_embeddings, data

question_embeddings, data = load_data()

# Load multilingual model for embedding generation
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Initialize the Google Sheets API client
def init_gspread():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if not creds_json:
        st.error("Google credentials environment variable is not set. Check Streamlit secrets.")
        raise Exception("Google credentials environment variable is not set.")
    creds_dict = json.loads(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key("1lAoDNeUYgwyYl0oYkRivu3hJSlrDrYgifMySC043Rw4").sheet1

sheet = init_gspread()

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1)
    return embeddings.cpu().numpy()

def translate_text(text, src_lang, dest_lang='en'):
    if src_lang == dest_lang:
        return text
    try:
        translated_text = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text

def find_closest_question_and_answer(query, src_lang):
    query_eng = translate_text(query, src_lang, 'en')
    query_emb = get_embedding(query_eng)
    similarities = {q: cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1)).flatten()[0] for q, emb in question_embeddings.items()}
    closest_question_eng = max(similarities, key=similarities.get)
    answer_eng = data[data['question'] == closest_question_eng]['answers'].iloc[0]
    closest_question = translate_text(closest_question_eng, 'en', src_lang)
    answer = translate_text(answer_eng, 'en', src_lang)
    return closest_question, answer

def generate_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    unique_filename = f"speech_{uuid.uuid4().hex}.mp3"
    tts.save(unique_filename)
    return unique_filename

def log_interaction_to_sheet(question, answer, detected_lang, actual_lang, relevance_score, correct_output, timestamp):
    data = [question, answer, detected_lang, actual_lang, relevance_score, correct_output, timestamp.strftime('%Y-%m-%d %H:%M:%S')]
    sheet.append_row(data)

def main():
    st.title("💬 Agricultural Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
        if "audio_file" in msg and os.path.exists(msg["audio_file"]):
            st.audio(msg["audio_file"])
    if question := st.chat_input("Ask me anything about agriculture:"):
        handle_conversation(question)

def handle_conversation(question):
    detected_lang = detect(question)
    src_lang_code = detected_lang if detected_lang in ['en', 'ml', 'te', 'hi', 'kn'] else 'en'
    closest_question, answer = find_closest_question_and_answer(question, src_lang_code)
    audio_file = generate_speech(answer, src_lang_code)
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    st.session_state["messages"].append({"role": "assistant", "content": answer, "audio_file": audio_file})
    st.chat_message("assistant").write(answer)
    if os.path.exists(audio_file):
        st.audio(audio_file)
    log_interaction_to_sheet(question, answer, detected_lang, src_lang_code, 5, True, datetime.datetime.now())

if __name__ == "__main__":
    main()
