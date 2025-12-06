from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
import uvicorn
import time
import uuid

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('public/index.html')

# Load model globally
MODEL_PATH = "kyr0/aidana-slm-mlx"
print(f"Loading model: {MODEL_PATH}")
model, tokenizer = load(MODEL_PATH)
print("Model loaded.")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[dict] = None

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    # Convert messages to format expected by tokenizer
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # Apply chat template to get the prompt string
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback if no chat template (basic concatenation)
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

    # Generate response
    # mlx_lm.generate(model, tokenizer, prompt, max_tokens, verbose, ...)
    # We map request params to generate args where possible
    # Create sampler
    # Create sampler
    sampler = make_sampler(request.temperature, request.top_p)
    
    # Create logits processors
    logits_processors = make_logits_processors(
        repetition_penalty=request.repetition_penalty
    ) if request.repetition_penalty else None

    if request.stream:
        async def event_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created = int(time.time())
            
            for response in generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens if request.max_tokens else 512,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False # Don't log to console during streaming
            ):
                # mlx_lm.generate returns a string when not streaming, but we want stream_generate
                # Wait, mlx_lm.generate calls stream_generate internally and yields if we iterate?
                # No, mlx_lm.generate returns a string. We need to call stream_generate directly.
                pass
            
            # Correct approach: call stream_generate directly
            from mlx_lm import stream_generate
            
            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens if request.max_tokens else 512,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                chunk = ChatCompletionResponse(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionResponseChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=response.text),
                            finish_reason=response.finish_reason
                        )
                    ],
                    usage=None
                )
                yield f"data: {chunk.json()}\n\n"
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    text = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        verbose=True, # Log to console
        max_tokens=request.max_tokens if request.max_tokens else 512,
        sampler=sampler,
        logits_processors=logits_processors
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": 0, # Not calculating exact tokens for speed
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
