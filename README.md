# Royal Spice Restaurant Chatbot (Advanced Hybrid RAG System)

This repository features a production-grade, state-of-the-art **Retrieval-Augmented Generation (RAG)** chatbot system designed for **Royal Spice Restaurant** (Hyderabad, India). 

It combines **Semantic Vector Search** and **Lexical Key Keyword Matching** running concurrently in parallel threads to deliver highly accurate, grounded, and extremely fast responses.

---

## 📸 Screenshots

<table>
<tr>
<td align="center">
<b>Restaurant Website</b><br><br>
<img src="royal spice restorent chatbot project/static/images/website.PNG" width="400">
</td>

<td align="center">
<b>AI Assistant Chatbot</b><br><br>
<img src="royal spice restorent chatbot project/static/images/AI_Bot.PNG" width="400">
</td>
</tr>
</table>
---

## 🌟 Key Highlights & Architecture

### 1. ⚡ Parallel Multithreaded Search (Concurrently executed)
To minimize latency and maximize speed, the chatbot executes two separate lookup mechanisms concurrently inside a `ThreadPoolExecutor`:
* **Thread 1: Optimized Keyword Search**: Matches query terms strictly against the keys of the JSON knowledge base (e.g. comparing terms to keys like `"restaurant menu with Dish Prices"`). This provides fast, high-accuracy lexical retrieval.
* **Thread 2: Vector similarity Search**: Converts the query and sub-queries into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2` and searches the local **Chroma DB** to check semantic relevance.

### 2. 🗂️ JSON-Structured Knowledge Base
The knowledge base has been converted from loose plain text to a valid, structured JSON file: [knowledge_base.json](knowledge_base.json).
* Chunks are indexed **directly** as key-value pairs. This keeps topics (like menus, timings, or capacity) completely self-contained, removing the risk of arbitrary page splitting.

### 3. 🔍 Separate Diagnostic Logs
When running the app, the terminal displays clear, categorized diagnostics separating what was found by each parallel thread:
* `[1. Keyword Search (Keys Only)] Chunks found`
* `[2. Vector Search (Semantic)] Chunks found`
* `[Response Mode]` (clearly showing that the answer is generated from the top 3 retrieved database chunks)

---

## 🖥️ Web App & Chatbot Interface

The project includes a full-stack FastAPI web application serving both the restaurant website and a premium chatbot interface:
* **Session-Isolated Chat History**: Uses cookie-based sessions (with unique UUIDs) to ensure that each user gets their own isolated chat history. The history only records the user's questions and the final assistant responses.
* **Glassmorphism UI**: Beautiful, modern dark-themed chat interface with custom typography (Inter font), smooth animations, live typing indicators, and full responsiveness (mobile-friendly).
* **Text-to-Speech (TTS)**: Interactive audio controls allowing users to read or listen to responses. Custom speaker controls accelerate speech for a more natural response reading.
* **Multi-Query Expansion**: Generates sub-questions from user inputs before executing retrieval, resolving phrasing variance.
* **Deduplication & Capping**: Chunks are deduplicated by key name and capped to a maximum of **3 unique chunks** in the context, preventing LLM token overhead.

---

## 🛠️ Required Packages & Dependencies

Make sure you have Python 3.10+ installed. Install the following libraries:

```bash
pip install fastapi uvicorn python-multipart jinja2 langchain langchain-community langchain-huggingface langchain-chroma chromadb langchain-groq sentence-transformers python-dotenv
```

---

## 📂 Project Directory Structure

```text
royal spice restorent chatbot project/
│
├── chroma_db/                  # Local directory containing persisted Chroma DB files
├── static/                     # Custom stylesheet (CSS), logo assets, and web scripts
├── templates/
│   ├── royal spice website.html # Restaurant website landing page
│   └── chatbot.html            # Premium glassmorphic chat interface
├── .env                        # Environment configuration file (Groq API Key)
├── app.py                      # FastAPI Server running the website and chatbot APIs
├── knowledge_base.json          # Structured restaurant JSON database
├── rag project.py              # Backend RAG script executing parallel search threads
└── README.md                   # Project documentation
```

---

## 🚀 Setup & Execution

### 1. Set Up Environment Variables
Create a `.env` file in the root of the project directory and insert your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*Get a free API key at [Groq Console](https://console.groq.com/).*

### 2. Run the Standalone CLI Chatbot
Run the RAG engine directly in your terminal to test parallel searches and terminal outputs:
```bash
python "rag project.py"
```

### 3. Run the FastAPI Web Application
Start the local server:
```bash
python app.py
```
* **Landing Page**: Navigate to `http://127.0.0.1:5000/` to view the restaurant website.
* **Chatbot Page**: Click the chat widget or go to `http://127.0.0.1:5000/AI_Bot` to open the chatbot interface. The terminal running `app.py` will print real-time parallel search engine logs as you converse!
