from fastapi import FastAPI, Request, Response, HTTPException, Cookie
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import importlib.util
import sys
import os
import uuid
import uvicorn

app = FastAPI()

# Mount static files (HTML/CSS/JS assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Dynamically import 'rag project' module because of the space in the filename
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, "rag project.py")
spec = importlib.util.spec_from_file_location("rag_project", module_path)
rag_project = importlib.util.module_from_spec(spec)
sys.modules["rag_project"] = rag_project
spec.loader.exec_module(rag_project)

# In-memory dictionary to store session-specific chat histories
sessions_history = {}

class ChatRequest(BaseModel):
    message: str

@app.get("/AI_Bot", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "chatbot.html")

@app.get("/", response_class=HTMLResponse)
async def website(request: Request):
    return templates.TemplateResponse(request, "royal spice website.html")

@app.post("/chat")
async def chat(request: Request, response: Response, chat_req: ChatRequest, session_id: Optional[str] = Cookie(None)):
    user_message = chat_req.message
    if not user_message:
        raise HTTPException(status_code=400, detail="No message provided")

    # Get or generate a session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        # Set session_id cookie (lasts for the browser session)
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    # Get or initialize history for this session
    if session_id not in sessions_history:
        sessions_history[session_id] = []

    chat_history = sessions_history[session_id]

    # Append user message to history
    chat_history.append({"role": "user", "content": user_message})
    try:
        # Get response from the RAG model
        answer, source_docs = rag_project.get_bot_response(user_message, chat_history[-10:])
        
        # Append bot response to history
        chat_history.append({"role": "assistant", "content": answer})
        
        return {"response": answer}
    except Exception as e:
        print(f"Error during RAG generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
