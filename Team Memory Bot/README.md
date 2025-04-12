# 🧠 Team Memory Bot

A scalable, memory-augmented Discord bot that captures, processes, and retrieves team conversations using embeddings, classification, and Retrieval-Augmented Generation (RAG). Ideal as a foundation for a full-blown AI assistant.

## 🚀 Features
✅ Real-time message capture and storage

🧠 NLP-based classification of relevant content

🔎 Context-aware retrieval using vector similarity

📦 Embedding with Sentence Transformers

🧠 Organized memory storage using ChromaDB

⚙️ Custom Discord commands and listeners

🧱 Modular, extensible, and production-ready architecture

## 🧰 Tech Stack

- **Language:** Python
- **Framework:** Discord.py (Bot API)
- **Embeddings:** sentence-transformers
- **Vector Store:** ChromaDB (SQLite-based)
- **Classifier:** Custom ML classifier (scikit-learn style)
- **Retrieval:** Semantic search (cosine similarity)
- **Config Mgmt:** Python dotenv (.env files)
- **Structure:** Clean, modular project layout

## 🛠️ Setup Instructions
### Clone the repository
```sh
git clone "https://github.com/yaminimanral/Data-Science-Projects.git"
cd Team Memory Bot
```

### Create a virtual environment
```sh
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

### Install dependencies
```sh
pip install -r requirements.txt
```

### Configure environment variables
- Copy .env.example to .env
- Fill in your Discord bot token and other required keys

### Run the bot
```sh
python src/main.py
```

## 📂 Project Structure
```
team-memory-bot/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── src/
│   ├── bot/                # Discord integration
│   ├── memory/             # DB & embedding logic
│   ├── preprocessing/      # NLP preprocessing
│   ├── query_engine/       # Retrieval engine (RAG)
│   └── config.py           # Configuration
├── chroma_db/              # Vector DB storage
```

## ✍️ Authors
- Rushi Karwankar - [GitHub](https://github.com/rkarwankar) - [LinkedIn](https://www.linkedin.com/in/rushikesh-karwankar/)
- Yamini Manral - [LinkedIn](https://www.linkedin.com/in/yaminimanral/)

Contributions welcome – feel free to submit a PR!

## 📄 License
This project is licensed under the [MIT License](https://github.com/yaminimanral/Data-Science-Projects/blob/main/LICENSE).