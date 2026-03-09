import os
import json
from huggingface_hub import snapshot_download
from huggingface_hub.utils import logging
logging.set_verbosity_info()

# compose_state.json 路徑
COMPOSE_STATE_PATH = os.path.join(
    os.path.dirname(__file__),
    "static",
    "app",
    "compose_state.json"
)

def download_model_func(file_path: str, service: str, model_name: str):
    """
    將模型加入列表，設定為使用中，並下載到 volume
    """
    try:
        # 1️⃣ 讀 / 寫 compose_state
        with open(file_path, "r", encoding="utf-8") as f:
            compose_state = json.load(f)

        model_list = compose_state.setdefault(service, {}).setdefault("model", {}).setdefault("list", [])

        # 檢查是否已存在
        if model_name in model_list:
            return {"success": False, "message": f"{model_name} 已經存在於列表中", "list": model_list}

        # 加入 list 並設定為使用中
        model_list.append(model_name)
        compose_state[service]["model"]["list"] = model_list
        compose_state[service]["model"]["use"] = model_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(compose_state, f, indent=2)

        # 2️⃣ Hugging Face 下載到 volume
        base_path = "/app/models"  # Volume 掛載點
        service_folder = os.path.join(base_path, service.lower())
        os.makedirs(service_folder, exist_ok=True)  # 建資料夾

        # 下載整個模型 repo
        snapshot_download(repo_id=model_name, cache_dir=service_folder, local_dir_use_symlinks=False)

        return {"success": True, "message": f"{model_name} 已加入列表並下載完成", "list": model_list}

    except Exception as e:
        return {"success": False, "error": str(e)}

def init_models():
    """
    啟動時檢查 compose_state.json，
    若 model list 為空但 use 有值則自動下載
    """
    if not os.path.exists(COMPOSE_STATE_PATH):
        print(f"[INIT] 找不到 compose_state.json: {COMPOSE_STATE_PATH}")
        return

    with open(COMPOSE_STATE_PATH, "r", encoding="utf-8") as f:
        compose_state = json.load(f)

    for service, data in compose_state.items():
        model_info = data.get("model", {})
        model_list = model_info.get("list", [])
        model_use = model_info.get("use")

        if not model_list and model_use:
            print(f"[INIT] {service} 模型列表為空，自動下載 {model_use}")
            download_model_func(COMPOSE_STATE_PATH, service, model_use)