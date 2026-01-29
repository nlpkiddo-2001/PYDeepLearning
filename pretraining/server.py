import os
import sys
import torch
import torch.nn.functional as F
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
from dataclasses import asdict

# Add src to path so we can import model
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from model import GPT, ModelConfig

# Configuration (Same as inference_pt.py)
CHECKPOINT_PATH = "/workspace/pretrain_slm/checkpoints/finetune/step_5000.pt"
TOKENIZER_PATH = os.path.join("/workspace/pretrain_slm/data/tokenizer.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Helper for Safe Globals
torch.serialization.add_safe_globals([ModelConfig])

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    load_inference_model()
    yield
    # Shutdown: Clean up if needed (nothing to do here)

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global Load
model = None
tokenizer = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None # If None, use model.args.max_seq_len

@app.get("/config")
async def get_config():
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)
    return JSONResponse(asdict(model.args))

def load_inference_model():
    global model, tokenizer
    from transformers import PreTrainedTokenizerFast

    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.eos_token = "<|endoftext|>"
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        config_obj = checkpoint.get("config", None)
        if config_obj is None:
            raise ValueError("No config found in checkpoint!")
        
        model = GPT(config_obj)
        state_dict = checkpoint.get("model", checkpoint)
        
        # Clean state dict
        clean_sd = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_sd[k[7:]] = v
            else:
                clean_sd[k] = v
        
        model.load_state_dict(clean_sd)
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def format_prompt(messages: List[Message]):
    formatted_parts = []
    for msg in messages:
        if msg.role == 'user':
            formatted_parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == 'assistant':
            formatted_parts.append(f"<|assistant|>\n{msg.content}")
        elif msg.role == 'system':
            formatted_parts.append(f"<|system|>\n{msg.content}")
            
    prompt = "\n".join(formatted_parts) + "\n<|assistant|>\n"
    return prompt

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def stream_generator(prompt, temperature=0.7, top_p=0.9, max_new_tokens=200):
    if model is None or tokenizer is None:
        yield "Error: Model not loaded."
        return

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        if input_tensor.shape[1] > model.args.max_seq_len:
            input_tensor = input_tensor[:, -model.args.max_seq_len:]

        with torch.no_grad():
            logits, _ = model(input_tensor)

        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        generated_token_id = next_token.item()
        
        if generated_token_id == tokenizer.eos_token_id:
            break
            
        token_text = tokenizer.decode([generated_token_id])
        yield token_text

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    prompt = format_prompt(request.messages)
    
    # Determine max tokens
    max_new_tokens = request.max_tokens
    if max_new_tokens is None:
        if model is not None:
             max_new_tokens = model.args.max_seq_len
        else:
             max_new_tokens = 2000

    async def response_generator():
        for token in stream_generator(prompt, request.temperature, request.top_p, max_new_tokens):
            yield token
            # Small delay to ensure smooth streaming feel if generation is too fast
            await asyncio.sleep(0.001) 

    return StreamingResponse(response_generator(), media_type="text/plain")

if __name__ == "__main__":
    # Disable reload to prevent accidental restarts on remote
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
