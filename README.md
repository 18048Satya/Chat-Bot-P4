# 💬 Interactive Chatbot with Streamlit and NLP
## Over view 
This is a chatbot application built using Streamlit, scikit-learn, and NLTK for natural language processing. The chatbot is powered by a RandomForestClassifier model that is trained on user intents defined in a JSON file.

---

## Features
-Natural Language Processing: The chatbot preprocesses user input using NLTK to filter out stopwords and tokenize the text.
-Model Training: The chatbot uses a RandomForestClassifier with TfidfVectorizer to classify user input into predefined intents.
-Conversation History: User conversations are logged and can be viewed in the 'History' page.
-Streamlit Interface: A simple and interactive web interface for chatting with the bot.

---

## Prerequisites
-Python (3.6 or above)
-Streamlit
-scikit-learn
-NLTK

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

---

## Usage
To run the chatbot application, execute the following command:
```bash
streamlit run app.py
```

Once the application is running, you can interact with the chatbot through the web interface. Type your message in the input box and press Enter to see the chatbot's response.

---

## Intents Data
The chatbot's behavior is defined by the `intents.json` file, which contains various tags, patterns, and responses. You can modify this file to add new intents or change existing ones.

---

## Conversation History
The chatbot saves the conversation history in a CSV file (`conversation_history.csv`). You can view past interactions by selecting the "Conversation History" option in the sidebar.

---

## Contributing
Contributions to this project are welcome! If you have suggestions for improvements or features, feel free to open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **NLTK** for natural language processing.
- **Scikit-learn** for machine learning algorithms.
- **Streamlit** for building the web interface.

---

Replace `<repository-url>` and `<repository-directory>` with the actual URL of your repository and the name of the directory where the project is located. Adjust any sections as necessary to better fit your project's specifics.
