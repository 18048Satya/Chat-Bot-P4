import os
import json
import datetime
import csv
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# File paths
INTENTS_FILE = 'intents.json'
CSV_FILE = 'conversation_history.csv'

# Load intents from the JSON file
def load_intents():
    with open(INTENTS_FILE, 'r') as file:
        return json.load(file)

# Preprocess text using NLTK
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  
    
    # Tokenize and convert to lowercase

    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Load conversation history from the CSV file
def load_history():
    history = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                history.append(row)
    return history

# Save conversation to the CSV file
def save_conversation(timestamp, user_message, bot_response):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp', 'user_message', 'bot_response'])
        if not file_exists:
            writer.writeheader()  # Write header if file is new
        writer.writerow({
            'timestamp': timestamp,
            'user_message': user_message,
            'bot_response': bot_response
        })

# Train the chatbot model
def train_model():
    intents = load_intents()
    training_sentences = []
    training_labels = []
    responses = {}

    # Prepare training data
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            processed_pattern = preprocess_text(pattern)  
    # Preprocess patterns
            training_sentences.append(processed_pattern)
            training_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']

    # Train model using TfidfVectorizer and RandomForestClassifier
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(training_sentences)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, training_labels)

    return model, vectorizer, responses

# Predict the response based on user input
def get_response(user_message, model, vectorizer, responses):
    processed_message = preprocess_text(user_message)  # Preprocess user message
    input_vector = vectorizer.transform([processed_message])
    intent = model.predict(input_vector)[0]
    return random.choice(responses[intent])

# Home page
def home_page():
    st.title("Welcome to the Chat App")
    st.write("This is an interactive chatbot app where you can communicate with an AI bot.")
    st.write("Navigate to different pages using the sidebar.")

# About page
def about_page():
    st.title("About the Chat App")
    st.write("""
    This chatbot uses intents stored in a JSON file to predict user queries and respond accordingly.
    Conversations are logged and displayed in the 'History' page. The app is powered by Streamlit,
    scikit-learn, and NLTK for natural language processing.
    """)

# History page
def history_page():
    st.title("Conversation History")
    history = load_history()
    if history:
        for entry in history:
            st.write(f"**[{entry['timestamp']}] User:** {entry['user_message']}")
            st.write(f"**[{entry['timestamp']}] Bot:** {entry['bot_response']}")
    else:
        st.write("No conversation history found.")

# Chat functionality and Home page 
def chat_page(model, vectorizer, responses):
    st.title("Chat Bot Using Basic Intents")
    user_message = st.text_input("You: ")
    if user_message:
        bot_response = get_response(user_message, model, vectorizer, responses)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_conversation(timestamp, user_message, bot_response)
        st.write(f"**Bot:** {bot_response}")

# Main app logic
def main():
    # Navigation using a selectbox
    page = st.sidebar.selectbox("", ["Home", "About", "History"])

    # Train the chatbot model
    model, vectorizer, responses = train_model()

    # Navigation logic
    if page == "Home":
        chat_page(model, vectorizer, responses)
    elif page == "About":
        about_page()
    elif page == "History":
        history_page()

if __name__ == "__main__":
    main()
