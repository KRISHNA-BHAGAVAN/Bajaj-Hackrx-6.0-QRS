Entity: An Intelligent Query & Reasoning System
Entity is a sophisticated, AI-powered system designed to understand and answer complex questions about a wide range of documents. It goes beyond simple Q&A by dynamically choosing the best strategy for a given task—either performing a deep search within a document's content or acting as an autonomous agent to interact with external web APIs to find the answer.

This project leverages the power of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and agentic workflows orchestrated by LangGraph to deliver precise, context-aware answers.

✨ Key Features
Multi-Format Document Parsing: Ingests and extracts text from various file types, including PDF, PPTX, XLSX, DOCX, and even images (JPG, PNG) using OCR.

Adaptive Reasoning Engine: Intelligently analyzes the document's content to decide whether to use a standard RAG pipeline for factual lookup or to deploy a ReAct (Reasoning and Acting) agent for tasks requiring external interaction.

Web-Aware Agent: The ReAct agent can autonomously use tools to scrape web pages and interact with APIs, following multi-step instructions found within a document to solve complex problems.

Robust Caching: Features a two-tier caching system for both FAISS vector indexes and text embeddings, dramatically speeding up subsequent requests for the same document.

Pluggable Models: Easily switch between different LLMs (OpenAI, Groq, Gemini) and embedding providers (NVIDIA, HuggingFace) through simple configuration changes.

Scalable & Asynchronous: Built with FastAPI and asynchronous processing, ensuring the system is fast, efficient, and can handle concurrent requests.

🏛️ Architecture
Entity is built on a modular, stateful graph managed by LangGraph. This architecture allows for a clear and maintainable flow of logic, where each node in the graph represents a specific processing step. The system's core is a conditional router that directs the workflow based on the nature of the input document.

graph TD
    A[API Request: URL + Questions] --> B{FastAPI Server};
    B --> C[LangGraph: Initialize State];
    C --> D{Cache Check: FAISS Index Exists?};
    D -- Yes --> F[Load Index from Cache];
    D -- No --> E[Process Document];
    E --> E1[1. Parse File: PDF, PPTX, OCR];
    E1 --> E2[2. Chunk Text];
    E2 --> E3[3. Embed Chunks];
    E3 --> E4[4. Create & Cache FAISS Index];
    E4 --> F;
    F --> G[Load Retriever];
    G --> H{Router: Analyze Content for API/URL Keywords};
    H -- No API/URL Found --> I[Standard RAG Pipeline];
    I --> I1[Retrieve Relevant Chunks];
    I1 --> I2[Generate Answer with LLM];
    I2 --> Z[Final Answer];
    H -- API/URL Found --> J[ReAct Agent Pipeline];
    J --> J1[Agent Executes Tools];
    J1 --> J2(Scrape Web / Call API);
    J2 --> J3[Reasoning Loop];
    J3 --> J1;
    J3 --> Z;
    Z --> B;

🚀 Getting Started
You can get the Entity server running in just a few steps. We recommend using Docker for the simplest and most reliable setup.

Prerequisites
Git

Docker and Docker Compose

An NVIDIA API Key and/or an OpenAI API Key

Method 1: Docker (Recommended)
Clone the repository:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

Configure Environment Variables:
Create a .env file by copying the example file.

cp .env.example .env

Now, open the .env file and add your API keys and authentication token.

# Default and Recommended
OPENAI_API_KEY=your-openai-api-key

# Default and Recommended embeddings through nvidia api
NVIDIA_API_KEY=your-nvidia-api-key

# Hackrx Team Authentication Token
AUTH_TOKEN=08fc8c10d11b09149c14f524da59050937f9875fbfa7190cebe26992162cd61b

# Optional: Add other keys like GROQ or GEMINI if you plan to use them

Build and Run the Docker Container:
This command builds the Docker image and starts the container. The -v flags create persistent volumes for the cache, so you don't lose your indexed data when the container stops.

docker build -t entity .

docker run \
  --rm \
  -p 8000:8000 \
  --env-file ./.env \
  -v faiss-cache-data:/app/faiss_cache \
  -v embed-cache-data:/app/embed_cache \
  entity

Ready! The API is now running and accessible at http://localhost:8000.

Method 2: Local Python Environment
Clone the repository and navigate into the directory.

Install System Dependencies:
This project requires Tesseract for OCR. On Debian/Ubuntu, you can install it with:

sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

Create a Python Virtual Environment:

python3 -m venv venv
source venv/bin/activate

Install Python Packages:

pip install -r requirements.txt

Configure Environment Variables as described in Step 2 of the Docker method.

Run the application:

uvicorn main:app --host 0.0.0.0 --port 8000

⚙️ Configuration
API Keys & Tokens (.env): All secrets and keys are managed in the .env file.

LLM and Embedding Models (config.py): You can easily swap out the models used for the RAG pipeline and the ReAct agent by commenting/uncommenting the relevant sections in config.py. This allows for great flexibility in balancing cost, speed, and capability.

🔌 API Usage
You can interact with the API using any HTTP client. Here is an example using curl.

Endpoint: POST /api/v1/hackrx/run

Headers:

Authorization: Bearer your-auth-token

Content-Type: application/json

Request Body:

{
  "documents": "URL_TO_YOUR_DOCUMENT_HERE",
  "questions": [
    "What is the primary conclusion of this document?",
    "Summarize the key financial figures for Q4."
  ]
}

Example curl command:

curl -X POST http://localhost:8000/api/v1/hackrx/run \
-H "Authorization: Bearer 08fc8c10d11b09149c14f524da59050937f9875fbfa7190cebe26992162cd61b" \
-H "Content-Type: application/json" \
-d '{
  "documents": "[https://hackrx.blob.core.windows.net/assets/Test%20/Test%20Case%20HackRx.pptx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A36%3A56Z&se=2026-08-05T18%3A36%3A00Z&sr=b&sp=r&sig=v3zSJ%2FKW4RhXaNNVTU9KQbX%2Bmo5dDEIzwaBzXCOicJM%3D](https://hackrx.blob.core.windows.net/assets/Test%20/Test%20Case%20HackRx.pptx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A36%3A56Z&se=2026-08-05T18%3A36%3A00Z&sr=b&sp=r&sig=v3zSJ%2FKW4RhXaNNVTU9KQbX%2Bmo5dDEIzwaBzXCOicJM%3D)",
  "questions": ["what is the name of the company?"]
}'

Successful Response (200 OK):

{
  "answers": [
    "The name of the company is 'Innovate Inc.'."
  ]
}

📂 Project Structure
.
├── faiss_cache/        # Stores cached FAISS indexes
├── embed_cache/        # Stores cached text embeddings
├── config.py           # Central configuration for models and constants
├── data_processing.py  # Handles document downloading, parsing, chunking, and embedding
├── document_parser.py  # Specific parsers for PPTX, XLSX, and images
├── graph_builder.py    # Defines the LangGraph state machine and workflow
├── llm_services.py     # Manages the RAG chain and LLM fallback logic
├── main.py             # FastAPI application entry point
├── models.py           # Pydantic models for API requests and responses
├── react_agent.py      # Implements the ReAct agent and its tools
├── utils.py            # Utility functions, including token verification
├── Dockerfile          # Instructions to build the Docker image
├── requirements.txt    # Python package dependencies
└── README.md           # This file

🤝 Contributing
Contributions are welcome! If you'd like to help improve Entity, please feel free to fork the repository, make your changes, and submit a pull request.

Fork the repository.

Create a new branch (git checkout -b feature/your-feature-name).

Make your changes and commit them (git commit -m 'Add some amazing feature').

Push to the branch (git push origin feature/your-feature-name).

Open a Pull Request.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.