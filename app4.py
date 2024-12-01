import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from langdetect import detect
from deep_translator import GoogleTranslator  # Using Deep Translate (GoogleTranslator)
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import pickle
import os
import uuid
import pandas as pd
import datetime

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

def get_embedding(text):
    """Generate an embedding for a given text."""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1)
    return embeddings

def translate_text(text, src_lang, dest_lang='en'):
    """Translate text between specified source and destination languages."""
    if src_lang == dest_lang:
        return text  # No translation needed
    try:
        translated_text = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
        return translated_text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text  # Return original text if translation fails

def find_closest_question_and_answer(query, src_lang):
    """Find the closest matching question and corresponding answer."""
    
    # Translate the query to English if needed
    query_eng = translate_text(query, src_lang, dest_lang='en')
    
    # Generate embedding and calculate cosine similarity
    query_emb = get_embedding(query_eng)
    similarities = {q: cosine_similarity(query_emb, emb).flatten()[0] for q, emb in question_embeddings.items()}
    closest_question_eng = max(similarities, key=similarities.get)
    
    # Retrieve the answer in English
    answer_eng = data[data['question'] == closest_question_eng]['answers'].iloc[0]
    
    # Translate back to the original language if necessary
    closest_question = translate_text(closest_question_eng, 'en', src_lang)
    answer = translate_text(answer_eng, 'en', src_lang)
    
    return closest_question, answer

def generate_speech(text, lang_code):
    """Generate a speech audio file for the given text."""
    tts = gTTS(text=text, lang=lang_code)
    unique_filename = f"speech_{uuid.uuid4().hex}.mp3"
    tts.save(unique_filename)
    return unique_filename

def save_interaction_to_csv(question, answer, detected_lang, actual_lang, relevance_score, correct_output, timestamp):
    """Log the interaction to a CSV file."""
    csv_path = r'conversation_logs.csv'
    df = pd.DataFrame([{
        'Question': question,
        'Answer': answer,
        'Detected Language': detected_lang,
        'Actual Language': actual_lang,
        'Relevance Score': relevance_score,
        'Correct Output': correct_output,
        'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

def main():
    st.title("💬 Agricultural Chatbot (Updated)")

    # Initialize session state for conversations
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages in chat-style format
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
        if "audio_file" in msg and os.path.exists(msg["audio_file"]):
            st.audio(msg["audio_file"])

    # Input box for user questions
    if question := st.chat_input("Ask me anything about agriculture:"):
        handle_conversation(question)

def handle_conversation(question):
    """Handle user input and generate a response."""
    detected_lang = detect(question)
    st.write(f"Detected language: {detected_lang}")  # Log the detected language
    src_lang_code = detected_lang

    # Find the closest question and corresponding answer
    closest_question, answer = find_closest_question_and_answer(question, src_lang_code)
    audio_file = generate_speech(answer, src_lang_code)

    # Append the user's message to the session state
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Append the bot's response to the session state, including the audio file
    st.session_state["messages"].append({"role": "assistant", "content": answer, "audio_file": audio_file})
    st.chat_message("assistant").write(answer)

    # Display the generated audio
    if os.path.exists(audio_file):
        st.audio(audio_file)

    # Save interaction to CSV
    save_interaction_to_csv(
        question, answer, detected_lang, detected_lang, 5, True, datetime.datetime.now()
    )

if __name__ == "__main__":
    main()
