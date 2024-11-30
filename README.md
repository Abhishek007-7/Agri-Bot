# 🌾 Agricultural Chatbot

This project is an interactive multilingual chatbot designed to provide answers to agricultural queries. It uses state-of-the-art NLP techniques, including embeddings from the `sentence-transformers` model, to find the closest matching question from a dataset and generate responses. The chatbot also supports language detection, translation, and speech synthesis for enhanced usability.

---

## 📋 Features

- **Multilingual Support**: Automatically detects the input language and provides answers in the same language using translation.
- **Question Similarity Matching**: Uses embeddings and cosine similarity to find the most relevant answer from the dataset.
- **Text-to-Speech**: Converts answers into audio for improved accessibility.
- **Logging**: Saves user interactions and their responses for future analysis.
- **Streamlit-Based UI**: A user-friendly web interface powered by Streamlit.

---

## 🛠️ Tech Stack

- **Python**: Programming language
- **Streamlit**: Web application framework
- **Transformers**: For embedding generation
- **Deep Translator**: For text translation
- **Google Text-to-Speech (gTTS)**: For speech synthesis
- **pandas**: For data handling
- **scikit-learn**: For cosine similarity computation
- **PyTorch**: Deep learning framework

---

## 📂 Project Structure

project/ │ ├── app.py # Main Streamlit application ├── datasets/ │ └── agri.csv # Dataset containing agricultural questions and answers ├── Saved_state/ │ └── embeddings.pkl # Preprocessed question embeddings ├── conversation_logs.csv # Logs of user interactions ├── requirements.txt # Python dependencies └── README.md # Project documentation

yaml
Copy code

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Internet connection for downloading models and using translation services

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/agricultural-chatbot.git
   cd agricultural-chatbot
Install dependencies:

bash
Copy code
pip install -r requirements.txt
(Optional) Install system dependencies for deployment: Create a packages.txt file with the following content:

plaintext
Copy code
libglib2.0-0
libsm6
libxrender1
libxext6
Then, use it in deployment platforms like Streamlit Cloud.

Preprocess the dataset: Run the preprocessing script to generate embeddings:

bash
Copy code
python preprocess.py
Launch the chatbot:

bash
Copy code
streamlit run app.py


## 🗂️ Dataset

The dataset (`agri.csv`) contains agricultural questions and their corresponding answers. Each row includes:
- `question`: The question text
- `answers`: The corresponding answer text

---

## 🤖 How It Works

1. **Input Processing**: 
   - Detects the language of the input query.
   - Translates it to English for processing.

2. **Question Matching**:
   - Generates embeddings for the query.
   - Computes cosine similarity with preprocessed question embeddings.

3. **Answer Generation**:
   - Retrieves the closest matching question's answer.
   - Translates the answer back to the user's original language.

4. **Text-to-Speech**:
   - Converts the response into an audio file for playback.

5. **Logging**:
   - Saves user interactions (question, answer, language, and timestamp) into a CSV file.

---

## 📝 Logging and Analytics

The chatbot logs the following information:
- User question
- Generated response
- Detected and actual language
- Relevance score
- Timestamp of the interaction

The logs are saved in `conversation_logs.csv` for later analysis.

---

## 📦 Deployment

1. **Local Deployment**:
   - Follow the installation steps and run the app locally.

2. **Streamlit Cloud**:
   - Push the repository to GitHub.
   - Connect your GitHub repo to [Streamlit Cloud](https://streamlit.io/cloud) and deploy.

3. **Other Platforms**:
   - Dockerize the app for deployment on platforms like AWS, Azure, or Google Cloud.

---

## 🔗 Example Use Cases

- Assisting farmers with agricultural queries.
- Educating students about agriculture.
- Building multilingual customer support tools.

---

## 💡 Future Improvements

- Add more robust datasets for agricultural knowledge.
- Implement voice recognition for spoken questions.
- Enhance the UI for better interactivity and user experience.

---

## 🧑‍💻 Author

- **Abhishek Madhu Vidya**  
  [GitHub](https://abhishek007-7.github.io/My_Portfolio/) | [LinkedIn](https://www.linkedin.com/in/abhishek-mv-2a1003109/)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
