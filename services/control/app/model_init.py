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
    將模型加入 list，並下載到 volume
    """
    try:
        # 1️⃣ 讀 compose_state
        with open(file_path, "r", encoding="utf-8") as f:
            compose_state = json.load(f)

        model_config = compose_state.setdefault(service, {}).setdefault("model", {})
        model_list = model_config.setdefault("list", [])
        model_download = model_config.setdefault("download", [])

        # 檢查是否已存在
        if model_name in model_list:
            return {
                "success": False,
                "message": f"{model_name} 已經存在於列表中",
                "list": model_list
            }

        # 2️⃣ 加入 list 並設定 use
        model_list.append(model_name)
        model_config["list"] = model_list

        # 先寫回（確保 state 先更新）
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(compose_state, f, indent=2)

        # 3️⃣ HuggingFace 下載
        base_path = "/app/models"
        service_folder = os.path.join(base_path, service.lower())
        os.makedirs(service_folder, exist_ok=True)

        success = download_model_three_method(service, model_name, service_folder)

        # 4️⃣ 下載成功才加入 download
        if success:
            if model_name not in model_download:
                model_download.append(model_name)
                model_config["download"] = model_download

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(compose_state, f, indent=2)

        return {
            "success": True,
            "message": f"{model_name} 已加入列表並下載完成",
            "list": model_list,
            "download": model_download
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    
def download_missing_model_func(file_path: str, service: str, model_name: str):
    """
    若模型存在於 list 但未出現在 download 中，補下載
    不修改 list 與 use
    """
    try:
        # 1️⃣ 讀 compose_state
        with open(file_path, "r", encoding="utf-8") as f:
            compose_state = json.load(f)

        model_config = compose_state.setdefault(service, {}).setdefault("model", {})
        model_list = model_config.setdefault("list", [])
        model_download = model_config.setdefault("download", [])

        # 檢查模型是否在 list
        if model_name not in model_list:
            return {
                "success": False,
                "message": f"{model_name} 不在模型列表中，略過下載",
                "list": model_list
            }

        # 已下載
        if model_name in model_download:
            return {
                "success": True,
                "message": f"{model_name} 已存在於 download 列表",
                "list": model_list
            }

        print(f"[AUTO DOWNLOAD] {service} 缺少模型 {model_name}，開始下載")

        # 2️⃣ HuggingFace 下載
        base_path = "/app/models"
        service_folder = os.path.join(base_path, service.lower())
        os.makedirs(service_folder, exist_ok=True)

        success = download_model_three_method(service, model_name, service_folder)

        if success:
            model_download.append(model_name)
            model_config["download"] = model_download

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(compose_state, f, indent=2)

            return {
                "success": True,
                "message": f"{model_name} 補下載完成",
                "download": model_download
            }

        else:
            return {
                "success": False,
                "message": f"{model_name} 下載失敗"
            }

    except Exception as e:
        return {"success": False, "error": str(e)}

def download_model_three_method(service, name, dir):
    try:
        if service in ["Chat", "Embedding", "STT"]:
            local_dir = os.path.join(dir, "hub")
            os.environ["HF_HOME"] = dir

            snapshot_download(
                repo_id=name,
                cache_dir=local_dir,
                local_dir_use_symlinks=False
            )

        elif service == "TTS":
            os.environ["HF_HOME"] = dir
            local_dir = os.path.join(dir, name)

            snapshot_download(
                repo_id=name,
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )

        return True

    except Exception as e:
        print(f"[DOWNLOAD ERROR] {service} {name} : {e}")
        return False

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
        model_download = model_info.get("download",[])

        if not model_list and model_use:
            print(f"[INIT] {service} 模型列表為空，自動下載 {model_use}")
            download_model_func(COMPOSE_STATE_PATH, service, model_use)
        
        for model_name in model_list:
            if model_name not in model_download:
                print(f"[INIT] {service} 模型 {model_name} 未下載，自動下載")
                download_missing_model_func(COMPOSE_STATE_PATH, service, model_name)
        