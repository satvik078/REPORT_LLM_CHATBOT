# 🛰️ Wildlife Drone Surveillance LLM Chatbot

A domain-specific AI chatbot built by fine-tuning a lightweight Large Language Model (LLM) on a custom dataset derived from a **Wildlife Surveillance Autonomous Drone System** project report.

The assistant provides **technical explanations, system insights, and conversational responses** through an interactive chat interface.

---

## 🚀 Project Highlights

* Fine-tuned **TinyLlama-1.1B-Chat** using LoRA (parameter-efficient training)
* Custom dataset (~500 project-specific Q&A samples)
* Training performed on **Google Colab free T4 GPU**
* Model merged and deployed for **local inference on Apple Silicon (MPS GPU)**
* Interactive **Streamlit chatbot UI**
* Hybrid chatbot behaviour:

  * casual responses for greetings / jokes
  * technical responses for project queries
* Fully **offline runnable after first setup**
* Total training cost: **$0**

---

## 🧠 System Architecture

```
Project Report
     ↓
Q&A Dataset Creation
     ↓
LoRA Fine-Tuning (Colab GPU)
     ↓
Adapter Merging
     ↓
Local Model Storage
     ↓
Streamlit Chat UI
     ↓
Hybrid AI Assistant
```

---

## 💬 Example Queries

* What is the role of LiDAR sensor?
* How does the alert system work?
* What are real-world deployment challenges?
* Explain edge computing in this project.
* Hi
* Tell me a joke

---

## 📂 Project Structure

```
REPORT_LLM_CHATBOT/
│
├── chat.py                 # Streamlit chatbot UI
├── download_model.py       # Script to download / prepare model
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ How to Run the Project Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/satvik078/REPORT_LLM_CHATBOT.git
cd REPORT_LLM_CHATBOT
```

---

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv ai_env
source ai_env/bin/activate        # Mac/Linux
```

Windows:

```bash
ai_env\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers streamlit accelerate sentencepiece peft
```

---

### 4️⃣ Download and Prepare Model

Run the provided script:

```bash
python download_model.py
```

This will:

* download base pretrained model
* load fine-tuned adapter
* merge weights
* save final model locally

After completion, a folder like:

```
mac_drone_model_merged/
```

will be created.

---

### 5️⃣ Launch Chatbot UI

```bash
streamlit run chat.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🧪 Chatbot Behaviour

The assistant uses **intent routing logic**:

* casual greetings → instant friendly reply
* technical project questions → LLM generated answer

This improves usability and reduces hallucination.

---

## 📊 Results

* Stable domain-specific responses
* Multi-sentence technical explanations
* Real-time local inference
* Reduced hallucination via prompt conditioning
* Lightweight deployment suitable for laptops

---

## 🔮 Future Improvements

* Retrieval-Augmented Generation (RAG) integration
* Better intent classification
* Voice-based interaction
* Web deployment for public demo
* Model quantization for faster inference

---

## 👨‍💻 Author

**Satvik Pandey**
AI • Robotics • Computer Vision • LLM Engineering

---

⭐ If you found this project useful, consider giving it a star!
