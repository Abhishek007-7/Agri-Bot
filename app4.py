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
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

# Load preprocessed embeddings and data
@st.cache_data
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
    sheet = client.open_by_key("1lAoDNeUYgwyYl0oYkRivu3hJSlrDrYgifMySC043Rw4").sheet1

    # Ensure headers are present for new metrics
    headers = [
        "Question", "Answer", "Detected Language", "Actual Language", 
        "Relevance Score", "Correct Output", "Response Time", 
        "Fallback Used", "Translation Correct", "Feedback Satisfactory", "Timestamp"
    ]
    if sheet.row_count == 0 or sheet.row_values(1) != headers:
        sheet.insert_row(headers, 1)
    
    return sheet

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

def find_closest_question_and_answer(query, src_lang, threshold=0.7):
    query_eng = translate_text(query, src_lang, 'en')
    query_emb = get_embedding(query_eng)
    similarities = {q: cosine_similarity(query_emb.reshape(1, -1), emb.reshape(1, -1)).flatten()[0] for q, emb in question_embeddings.items()}
    
    closest_question_eng, max_similarity = max(similarities.items(), key=lambda x: x[1])
    
    if max_similarity < threshold:
        return None, None, float(max_similarity)
    
    answer_eng = data[data['question'] == closest_question_eng]['answers'].iloc[0]
    closest_question = translate_text(closest_question_eng, 'en', src_lang)
    answer = translate_text(answer_eng, 'en', src_lang)
    return closest_question, answer, float(max_similarity)

def generate_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    unique_filename = f"speech_{uuid.uuid4().hex}.mp3"
    tts.save(unique_filename)
    return unique_filename

def log_interaction_to_sheet(question, answer, detected_lang, actual_lang, relevance_score, correct_output, response_time, fallback_used, translation_correct, feedback_satisfactory, timestamp):
    # Ensure all values are serializable
    answer = answer if answer is not None else "N/A"
    relevance_score = float(relevance_score)  # Convert float32 to Python float
    correct_output = str(correct_output)  # Convert boolean to string
    translation_correct = str(translation_correct) if translation_correct is not None else "N/A"
    feedback_satisfactory = str(feedback_satisfactory) if feedback_satisfactory is not None else "N/A"
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')  # Format datetime as string

    data = [
        question, answer, detected_lang, actual_lang, relevance_score,
        correct_output, response_time, fallback_used, translation_correct,
        feedback_satisfactory, timestamp_str
    ]
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
    start_time = time.time()  # Record start time for response time calculation
    detected_lang = detect(question)
    src_lang_code = detected_lang if detected_lang in ['en', 'ml', 'te', 'hi', 'kn'] else 'en'
    closest_question, answer, similarity = find_closest_question_and_answer(question, src_lang_code)

    fallback_used = False
    if closest_question is None:
        answer = "I'm sorry, I couldn't find an answer to your question in my dataset."
        fallback_used = True

    audio_file = generate_speech(answer, src_lang_code)
    response_time = round(time.time() - start_time, 2)  # Calculate response time

    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)
    st.session_state["messages"].append({"role": "assistant", "content": answer, "audio_file": audio_file})
    st.chat_message("assistant").write(answer)
    if os.path.exists(audio_file):
        st.audio(audio_file)

    # Collect feedback from the user
    st.write("### Feedback:")
    feedback_satisfactory = st.radio(
        "Was the answer satisfactory?",
        options=["Select", "Yes", "No"],  # Add a placeholder
        key=f"feedback_satisfactory_{uuid.uuid4().hex}"
    )
    translation_correct = None
    if detected_lang != "en":
        translation_correct = st.radio(
            "Was the translation done correctly?",
            options=["Select", "Yes", "No"],  # Add a placeholder
            key=f"translation_correct_{uuid.uuid4().hex}"
        )

    # Ensure no "Select" is logged; use None for unselected
    feedback_satisfactory = None if feedback_satisfactory == "Select" else feedback_satisfactory
    translation_correct = None if translation_correct == "Select" else translation_correct

    # Log interaction
    log_interaction_to_sheet(
        question, answer, detected_lang, src_lang_code, similarity, not fallback_used,
        response_time, fallback_used, translation_correct, feedback_satisfactory,
        datetime.datetime.now()
    )

if __name__ == "__main__":
    main()
