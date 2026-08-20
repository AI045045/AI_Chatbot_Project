# AI_Chatbot_Project
<!-- AI Legal Chatbot - Project Documentation -->

# AI Local Legal Chatbot

A privacy-first, local-only Legal Document QA and Chat system built with **Streamlit**, **LangChain**, and **Ollama**. All document text extraction, vector embeddings, and LLM inference run strictly on your local machine.

---

## Features

- **Local & Private:** No document data or queries are sent to external APIs or third-party servers.
- **Support for PDF and DOCX:** Extract text, split into overlapping chunks, and perform local semantic search.
- **Vector Search (FAISS):** Fast in-memory similarity indexing without complex C++ compilation issues.
- **Citations:** Every generated response points directly to the document sections used for context.
- **User Authentication:** Simple local SQLite database storing bcrypt-hashed user and admin credentials.
- **Customizable Models:** Easily swap embedding and chat models via the UI.

---

## Getting Started

### 1. Prerequisites

- **Python 3.9+**
- **Ollama:** A local model runner. Download and install it from [Ollama's website](https://ollama.com).

### 2. Ollama Configuration

Start Ollama and pull the default models used by the app:

```bash
# Pull the embedding model (used to index documents)
ollama pull nomic-embed-text

# Pull the LLM chat model (used to answer queries)
ollama pull llama3:8b
```

### 3. Running the Chatbot

1. **Activate the Virtual Environment:**
   ```bash
   source .venv/bin/activate
   ```
2. **Run the Streamlit Web Application:**
   ```bash
   streamlit run "ai legal chatbot/app.py"
   ```
3. **Access the Chatbot:**
   Open your browser and navigate to `http://localhost:8501`.

---

## Containerization & Running with Docker Desktop

This project is containerized. You can run the chatbot inside a Docker container while communicating with Ollama running locally on your host Mac.

### 1. Prerequisites
- Install **Docker Desktop** from [Docker's official website](https://www.docker.com/products/docker-desktop/).
- Ensure **Ollama** is running locally on your host machine with the default models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3:8b
  ```

### 2. Configure Ollama for Host Access (Mac)
By default, Ollama only listens to local loopback requests (`127.0.0.1`). Because the application runs inside Docker, it must communicate with Ollama on the host via `host.docker.internal`.
1. Close Ollama if it is running in your Mac menu bar.
2. Start Ollama from your terminal with the environment variable set to bind to all interfaces:
   ```bash
   OLLAMA_HOST=0.0.0.0 ollama serve
   ```
   *(Keep this terminal window open to keep Ollama running).*

### 3. Build and Run with Docker Compose (CLI)
From the root of the project directory, run:
```bash
# Build and start the container
docker compose up --build
```
Once started, the application will be accessible in your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

### 4. Running and Managing via Docker Desktop (UI)
Once you have initialized the project, or to manage it graphically:

1. **Open Docker Desktop** on your Mac.
2. Navigate to the **Containers** tab from the left sidebar.
3. You will see a project stack named `ai-legal-chatbot` (or matching your project directory name).
4. **Control actions:**
   - Click the **Start (Play)** button to run the chatbot.
   - Click the **Stop (Square)** button to shut it down.
   - Click the **Restart** button to reload.
5. **View Logs and Files:**
   - Click on the container name `ai-legal-chatbot` to inspect live container logs.
   - Use the **Terminal** tab to run CLI commands directly inside the container (e.g. testing host connection via `curl http://host.docker.internal:11434`).
6. **Files and Database Persistence:**
   - The SQLite database (`users.db`) and uploaded files are mapped to persistent Docker volumes. Your user sessions and data will remain intact when stopping or rebuilding containers.

---

