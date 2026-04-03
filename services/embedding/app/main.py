import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Union
from fastapi.responses import JSONResponse
from transformers import AutoConfig, AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union, Optional

try:
    from auto_gptq import AutoGPTQForCausalLM
    GPTQ_AVAILABLE = True
except:
    GPTQ_AVAILABLE = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["HF_HOME"] = "/models"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/models/ktoprakucar/gte-Qwen2-1.5B-instruct-Q8-GPTQ"
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

def encode(texts: Union[str, List[str]], target_dim: int = 0):
    """
    將文字轉成向量
    :param texts: 單字串或字串列表
    :param target_dim: 如果 >0，輸出會補零或截斷到該維度；0 或 None 表示不補零
    """
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

    embeddings = embeddings.cpu().numpy()

    # 如果 target_dim > 0 才補零或截斷
    if target_dim and embeddings.shape[1] != target_dim:
        if embeddings.shape[1] < target_dim:
            padding = np.zeros((embeddings.shape[0], target_dim - embeddings.shape[1]), dtype=embeddings.dtype)
            embeddings = np.concatenate([embeddings, padding], axis=1)
        else:
            embeddings = embeddings[:, :target_dim]

    return embeddings

class TextInput(BaseModel):
    text: str
    dimensions: Optional[int] = None  # 可以選擇性輸入維度

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    dimensions: Optional[int] = None  # 可選

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
        dim = request.dimensions or 0
        vectors = encode(texts, target_dim=dim)

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
        dim = input.dimensions or 0
        vectors = encode(input.text, target_dim=dim)
        embedding = vectors[0]

        return {
            "embedding": embedding.tolist(),
            "dimension": embedding.shape[0],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return Response(status_code=200, content="OK")