# Speaking Practice Chatbot

**Speaking Practice Chatbot** is an AI-powered English tutor that helps learners improve their speaking skills through real-time voice conversations. Using **speech recognition**, **RAG (retrieval-augmented generation)**, and **speech synthesis**, it provides natural, interactive practice with personalized feedback via the friendly EnglishBuddy persona.
---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Future Development](#future-development)

---

## Features

- 🎙️ **Real-time Speech Processing**: The chatbot accepts real-time audio input from the user and converts the speech into text using a high-quality speech-to-text (STT) system.
- 📖 **Knowledge Augmentation**: RAG pipeline retrieves relevant information from a curated knowledge base to generate accurate, context-aware responses.
- 🗣️ **Text-to-Speech (TTS) with Voice Cloning**: The chatbot converts the generated text response back into natural, expressive speech using text-to-speech synthesis. The TTS module supports voice cloning, enabling the chatbot to mimic specific speaker voices for a personalized experience.
- ⚙️ **Configurable Models**: Flexible configuration via `config.yaml` for model sizes and parameters.

---

## Installation

To set up the Speaking Practice Chatbot locally:

1. **Clone the repository:**

```bash
git clone https://github.com/DoubleC04/speaking_practice_chatbot.git
cd speaking_practice_chatbot/chatbot
```

2. **Install Ollama:**  
Download and install Ollama to run the `english_buddy` model locally.

- Visit Ollama's official website and follow the installation instructions for your OS.
- After installation, run the model:

```bash
ollama create english_buddy -f Modelfile
```

3. **Install dependencies:**  

```bash
pip install -r requirements.txt
```

4. **Configure models:**  
Update `config.yaml` to set model devices (`cpu` or `cuda`):

```yaml
stt:
  device: "cuda"  # or "cpu"
```
> 💡 **Note for CPU Users**  
If you're running the chatbot on a CPU-only environment and encounter issues with `bitsandbytes`, you need to modify the import statement in the file `moshi/utils/quantize.py` to prevent errors.

### 🔧 How to fix

**Replace this line:**

```python
import bitsandbytes as bnb
```
with:

```python
try:
    import bitsandbytes as bnb
except Exception:
    bnb = None
```


5. **Initialize vector store:**

```bash
python -m src/scripts/initialize_vectorstore.py
```
> 💡 **Note:**  
You can add more learning materials by placing `.csv` and `.pdf` files into the `data/csv` and `data/pdf` directories, respectively. The vector store will index these documents for use in conversation.

**Currently, the project only supports CSV and PDF file formats.**

6. **Run the application:**

```bash
python -m src/scripts/run_chatbot.py
```

---

## Future Development
In the future, the chatbot will be enhanced with the following features:
- 🌐 **Web Deployment**: Deploy the chatbot on the web for easier accessibility and a wider user base.
- 📁 **Expanded File Format Support**: Add support for a wider variety of document types.
- 🏆 **User Proficiency Assessment**: Automatically evaluate learners’ English proficiency level through interactive tests and conversation analysis.
- 🎓 **Exam Preparation Support**: Provide targeted practice and guidance for popular English exams such as IELTS, TOEIC.
- 🌍 **Multilingual Support**: Expand the chatbot’s capabilities to support multiple languages, enabling learners from diverse backgrounds to practice and improve in their native languages as well as in English.


**Happy English practicing with EnglishBuddy! 🇬🇧🧠**
