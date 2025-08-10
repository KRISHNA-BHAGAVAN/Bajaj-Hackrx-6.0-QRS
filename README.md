# Entity: Intelligent Query & Reasoning System

Entity is a **sophisticated AI-powered platform** that understands and answers complex questions from a variety of documents.  
It goes far beyond simple Q&A by **dynamically selecting the optimal reasoning strategy**—either performing deep semantic search within document content or acting as an **autonomous agent** that can interact with external APIs and the web.

Built with **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **agentic workflows powered by LangGraph**, Entity delivers **precise, context-aware answers** in real-time.

---

## ✨ Key Features

- **Multi-Format Document Parsing**  
  Parse and extract text from PDFs, PPTX, XLSX, DOCX, and even images (`JPG`, `PNG`) via OCR.

- **Adaptive Reasoning Engine**  
  Automatically chooses between:
  - **Standard RAG Pipeline** for factual lookups.
  - **ReAct Agent** for reasoning + external API/web interactions.

- **Web-Aware Autonomous Agent**  
  Executes tools to scrape pages, call APIs, and follow multi-step instructions embedded in documents.

- **Two-Tier Caching System**  
  Speeds up repeated queries with **FAISS vector index caching** and **embedding caching**.

- **Pluggable Model Support**  
  Swap between OpenAI, Groq, Gemini, NVIDIA, HuggingFace embeddings with simple config changes.

- **Scalable & Async**  
  Built with **FastAPI** + asynchronous processing for high concurrency and performance.

---

## 🏛 Architecture

Entity is implemented as a **stateful LangGraph workflow**—each node represents a specific processing stage, with a conditional router deciding the path based on document type and query intent.

<p align="center">
  <img src="workflow.png" alt="Entity Architecture" width="700"/>
</p>

---

## 🚀 Getting Started

We recommend running Entity with **Docker** for the easiest and most consistent setup.

### **Prerequisites**

* Git
* Docker & Docker Compose
* API keys (NVIDIA and/or OpenAI)

---

### **Method 1 — Docker (Recommended)**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Set Up Environment Variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   NVIDIA_API_KEY=your-nvidia-api-key
   AUTH_TOKEN=08fc8c10d11b09149c14f524da59050937f9875fbfa7190cebe26992162cd61b
   ```

3. **Build & Run the Image**

   ```bash
   docker build -t entity .
   ```
   
4. **Run the Container**
   
   ```bash
   docker run \
     --rm \
     -p 8000:8000 \
     --env-file ./.env \
     -v faiss-cache-data:/app/faiss_cache \
     -v embed-cache-data:/app/embed_cache \
     entity
   ```

   Entity will be available at:
   **[http://localhost:8000](http://localhost:8000)**

---

### **Method 2 — Local Python Environment**

1. **Clone & Enter Directory**

   ```bash
   git clone https://github.com/KRISHNA-BHAGAVAN/Bajaj-Hackrx-6.0-Entity.git
   cd Bajaj-Hackrx-6.0-Entity
   ```

2. **Install OCR Dependency**

   ```bash
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
   ```

3. **Create Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python Packages**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run Application**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

---

## ⚙️ Configuration

* **API Keys** → `.env` file
* **LLM & Embedding Models** → `config.py`
  Swap models by commenting/uncommenting in the config.

---

## 🔌 API Usage

**Endpoint:**
`POST /api/v1/hackrx/run`

**Headers:**

```http
Authorization: Bearer your-auth-token
Content-Type: application/json
```

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/v1/hackrx/run \
-H "Authorization: Bearer 08fc8c10d11b09149c14f524da59050937f9875fbfa7190cebe26992162cd61b" \
-H "Content-Type: application/json" \
-d '{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is the company name?"]
}'
```

**Example Response:**

```json
{
  "answers": [
    "The name of the company is 'Innovate Inc.'."
  ]
}
```

---

## 📂 Project Structure

```
.
├── faiss_cache/         # Cached FAISS indexes
├── embed_cache/         # Cached embeddings
├── config.py            # Config for models & constants
├── data_processing.py   # Document download, parsing, embedding
├── document_parser.py   # File-type specific parsers
├── graph_builder.py     # LangGraph workflow definition
├── llm_services.py      # RAG chain & LLM fallback logic
├── main.py              # FastAPI entry point
├── models.py            # Pydantic request/response models
├── react_agent.py       # ReAct agent implementation
├── utils.py             # Utility functions
├── Dockerfile           # Docker build instructions
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add amazing feature"
   ```
4. Push to branch:

   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request

---

## 📜 License

Licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
