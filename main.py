# main.py
import uvicorn
import logging
import warnings
from fastapi import FastAPI, Depends, HTTPException, status

# Local module imports
from models import QueryRequest, QueryResponse
from utils import verify_token
from graph_builder import jarvis

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Logging Setup ---
LOG_FILE = "query_logs.log"
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="An LLM-powered system to answer questions about large documents.",
    version="1.0.0"
)

# --- API Endpoints ---
@app.post(
    "/api/v1/hackrx/run",
    response_model=QueryResponse,
    summary="Run a submission to query a document",
    dependencies=[Depends(verify_token)]
)

async def run_submission(request: QueryRequest):
    """
    This endpoint processes a document from a URL and answers a list of questions about it.

    - **documents**: URL to the document (PDF, PPTX, XLSX, etc.).
    - **questions**: A list of natural language questions.
    """
    logging.info(f"Received Document: {request.documents}")
    logging.info(f"Received Questions: {request.questions}")

    try:
        final_state = await jarvis.ainvoke({"doc_url": request.documents, "questions": request.questions})
        answers = final_state.get("answers", ["No answer could be generated."])
        
        logging.info("Generated Answers:")
        for i, a in enumerate(answers, start=1):
            logging.info(f"  A{i}: {a}\n" + "-"*60)
            
        return QueryResponse(answers=answers)

    except Exception as e:
        logging.exception("An unexpected error occurred")
        raise HTTPException(500, detail="Internal Server Error")

@app.get("/", summary="Health Check")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Query-Retrieval System is running."}


# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# To expose it as public url run the below command in a new terminal after running the above command
# ngrok http 8000