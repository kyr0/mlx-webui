from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models import cache as mlx_cache
import mlx.core as mx
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
print(f"Loading chat model: {MODEL_PATH}")
model, tokenizer = load(MODEL_PATH)
print("Chat model loaded.")

EMBEDDING_MODEL_PATH = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
print(f"Loading embedding model: {EMBEDDING_MODEL_PATH}")
emb_model, emb_tokenizer = load(EMBEDDING_MODEL_PATH)
print("Embedding model loaded.")


# Warmup
print("Warming up chat model...")
generate(model, tokenizer, prompt="Hello", max_tokens=1, verbose=False)
print("Chat model warmed up.")

print("Warming up embedding model...")
# Run a dummy forward pass
dummy_input = mx.array([emb_tokenizer.encode("Hello")])
emb_model.model(dummy_input)
print("Embedding model warmed up.")

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

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = "text-embedding-default"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict

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

    # Tokenize the prompt first to handle caching
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_tokens_list = prompt_tokens

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
            
            
            # We need to capture the generated tokens to update previous_tokens
            generated_tokens = []
            
            from mlx_lm import stream_generate
            
            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens if request.max_tokens else 512,
                sampler=sampler,
                logits_processors=logits_processors
            ):
                generated_tokens.append(response.token)
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
        verbose=True, 
        max_tokens=request.max_tokens if request.max_tokens else 512,
        sampler=sampler,
        logits_processors=logits_processors
    )
    
    if text is None:
        text = ""
    
    generated_tokens = tokenizer.encode(text, add_special_tokens=False)

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
            "prompt_tokens": len(prompt_tokens_list),
            "completion_tokens": len(generated_tokens),
            "total_tokens": len(prompt_tokens_list) + len(generated_tokens)
        }
    )

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    inputs = request.input
    if isinstance(inputs, str):
        inputs = [inputs]
    
    data = []
    total_tokens = 0
    
    for i, text in enumerate(inputs):
        tokens = emb_tokenizer.encode(text)
        total_tokens += len(tokens)
        input_ids = mx.array([tokens])
        
        # Forward pass through the backbone to get hidden states
        # Qwen3Model returns hidden states directly
        outputs = emb_model.model(input_ids)
        
        # Extract last hidden state (last token)
        # Shape: (1, seq_len, hidden_dim)
        # We want the last token: (1, 1, hidden_dim) -> (hidden_dim,)
        last_hidden_state = outputs[:, -1, :]
        
        # Normalize (L2)
        # norm = sqrt(sum(x^2))
        norm = mx.linalg.norm(last_hidden_state, ord=2, axis=-1, keepdims=True)
        normalized_embedding = last_hidden_state / norm
        
        # Convert to list
        embedding_list = normalized_embedding[0].tolist()
        
        data.append(EmbeddingData(
            embedding=embedding_list,
            index=i
        ))
        
    return EmbeddingResponse(
        data=data,
        model=request.model or "text-embedding-default",
        usage={
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
