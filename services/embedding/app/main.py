import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from fastapi.responses import JSONResponse
from transformers import AutoConfig, AutoTokenizer, AutoModel

try:
    from auto_gptq import AutoGPTQForCausalLM
    GPTQ_AVAILABLE = True
except:
    GPTQ_AVAILABLE = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
os.environ["TORCH_HOME"] = "/root/.cache/huggingface"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "ktoprakucar/gte-Qwen2-1.5B-instruct-Q8-GPTQ"
)

app = FastAPI(title="Universal Embedding API")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    if GPTQ_AVAILABLE and os.path.exists(os.path.join(MODEL_PATH, "quantize_config.json")):
        print(f"Loading GPTQ model {MODEL_PATH}...")
        model = AutoGPTQForCausalLM.from_quantized(
            MODEL_PATH,
            device=device,
            use_safetensors=True,
            trust_remote_code=True
        )
        is_gptq = True
    else:
        print(f"Loading standard HF model {MODEL_PATH}...")
        config = AutoConfig.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH, config=config)
        model.to(device)
        is_gptq = False

    model.eval()
    print("Model loaded on:", device)

except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def encode(texts: Union[str, List[str]]):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        if is_gptq:
            outputs = model.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  
        else:
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        embeddings = mean_pooling(hidden_states, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()

class TextInput(BaseModel):
    text: str

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class EmbeddingObject(BaseModel):
    object: str
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str
    data: List[EmbeddingObject]
    model: str

@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):

    if not request.input:
        return openai_error("`input` is required", param="input")

    try:
        texts = request.input if isinstance(request.input, list) else [request.input]
        vectors = encode(texts)

        data = [
            EmbeddingObject(
                object="embedding",
                embedding=vec.tolist(),
                index=i
            )
            for i, vec in enumerate(vectors)
        ]

        return EmbeddingResponse(
            object="list",
            data=data,
            model=request.model
        )

    except Exception as e:
        return openai_error(str(e), type_="internal_server_error", status_code=500)

def openai_error(message, type_="invalid_request_error", param=None, code=None, status_code=400):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": type_,
                "param": param,
                "code": code
            }
        }
    )

@app.post("/embed")
def embed(input: TextInput):
    try:
        vectors = encode(input.text)

        embedding = vectors[0]

        return {
            "embedding": embedding.tolist(),
            "dimension": embedding.shape[0],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))