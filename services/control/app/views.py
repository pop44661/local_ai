import os
import json
import subprocess

from subprocess import Popen, PIPE
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# API

COMPOSE_STATE_PATH = os.path.join(os.path.dirname(__file__), 'static', 'app', 'compose_state.json')

def index(request):
    # SPA 主頁

    with open(COMPOSE_STATE_PATH, 'r', encoding='utf-8') as f:
        compose_state = json.load(f)

    sync_container_status(COMPOSE_STATE_PATH)

    return render(request, "index.html", {
        "compose_state": json.dumps(compose_state)  # ⚡ 轉成合法 JSON 字串
    })

@csrf_exempt
def compose_state(request):
    if request.method != "POST":
        return JsonResponse({"error": "只支援 POST 方法"}, status=405)

    try:
        data = json.loads(request.body)

        service = data.get("service")
        model = data.get("model")
        exists = data.get("exists")
        running = data.get("running")

        result = update_compose_state(
            COMPOSE_STATE_PATH,
            service,
            model=model,
            exists=exists,
            running=running
        )

        if result["success"]:
            return JsonResponse(result)

        return JsonResponse(result, status=400)

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@csrf_exempt
def container_status(request):

    try:
        result = sync_container_status(COMPOSE_STATE_PATH)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse(
            {"success": False, "error": str(e)},
            status=500
        )

def gpu_info(request):

    result = get_gpu_info()

    if result.get("success"):
        return JsonResponse(result)

    return JsonResponse(result, status=500)

@csrf_exempt
def container_gpu_stats(request):
    try:
        result = get_container_gpu_stats(COMPOSE_STATE_PATH)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


@csrf_exempt
def restart_service_api(request):
    service = request.GET.get("service")
    result = restart_service(service)

    status = 200 if result.get("success") else 400
    if "error" in result and status == 200:
        status = 500  # 如果有 error 但 success = False

    return JsonResponse(result, status=status)
    

def docker_up(request):
    service = request.GET.get("service", "")
    result = start_containers(service)
    return JsonResponse(result)


def docker_down(request):
    service = request.GET.get("service", "")
    result = stop_containers(service)
    return JsonResponse(result)

def get_models_api(request):
    service = request.GET.get("service")
    if not service:
        return JsonResponse({"success": False, "error": "service is required"}, status=400)
    result = get_models_func(COMPOSE_STATE_PATH, service)
    status = 200 if result.get("success") else 500
    return JsonResponse(result, status=status)

@csrf_exempt
def download_model_api(request):
    data = json.loads(request.body)
    service = data.get("service")
    model_name = data.get("model_name")
    if not service or not model_name:
        return JsonResponse({"success": False, "error": "service and model_name required"}, status=400)
    
    result = download_model_func(COMPOSE_STATE_PATH, service, model_name)
    status = 200 if result.get("success") else 500
    return JsonResponse(result, status=status)

@csrf_exempt
def delete_model_api(request):
    data = json.loads(request.body)
    service = data.get("service")
    model_name = data.get("model_name")
    if not service or not model_name:
        return JsonResponse({"success": False, "error": "service and model_name required"}, status=400)

    result = delete_model_func(COMPOSE_STATE_PATH, service, model_name)
    status = 200 if result.get("success") else 500
    return JsonResponse(result, status=status)

@csrf_exempt
def select_model_api(request):
    """
    API: 將指定模型設為使用中
    """
    try:
        data = json.loads(request.body)
        service = data.get("service")
        model_name = data.get("model_name")

        if not service or not model_name:
            return JsonResponse({"success": False, "error": "service 和 model_name 都是必填"}, status=400)

        result = select_model_func(COMPOSE_STATE_PATH, service, model_name)
        status = 200 if result.get("success") else 400
        if "error" in result and result.get("success"):
            status = 500  # 如果 success=True 但仍有 error

        return JsonResponse(result, status=status)

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

@csrf_exempt
def generate_license_api(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required"}, status=400)

    user = request.POST.get("user")
    days = request.POST.get("days", 30)
    features = request.POST.get("features", '[]')
    private_key_file = request.FILES.get("private_key_file")

    if not user or not private_key_file:
        return JsonResponse({"success": False, "error": "Missing parameters"}, status=400)

    try:
        result = generate_license_request(private_key_file, user, days, features)
    except requests.RequestException as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

    return JsonResponse({"success": True, "upload_response": result})


@csrf_exempt
def upload_license_api(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST required"}, status=400)

    license_file = request.FILES.get("file")
    if not license_file:
        return JsonResponse({"success": False, "error": "No file uploaded"}, status=400)

    try:
        result = upload_license_request(license_file)
    except requests.RequestException as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)

    return JsonResponse({"success": True, "upload_response": result})













# Function
import docker
client = docker.from_env()
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import logging
logging.set_verbosity_info()
import shutil


def start_containers(services: str):
    """
    一次啟動多個 container，services 可以是 "chat embedding tts"
    """
    service_list = services.split()
    results = []

    # 取得所有容器
    all_containers = {c.name: c for c in client.containers.list(all=True)}

    for s in service_list:
        container = all_containers.get(s)
        if container:
            try:
                container.start()
                results.append({"service": s, "success": True, "message": f"{s} 已啟動"})
            except Exception as e:
                results.append({"service": s, "success": False, "error": str(e)})
        else:
            results.append({"service": s, "success": False, "error": f"{s} 不存在"})

    return {"results": results}


def stop_containers(services: str):
    """
    一次停止多個 container，services 可以是 "chat embedding tts"
    """
    service_list = services.split()
    results = []

    # 取得所有容器
    all_containers = {c.name: c for c in client.containers.list(all=True)}

    for s in service_list:
        container = all_containers.get(s)
        if container:
            try:
                container.stop()
                results.append({"service": s, "success": True, "message": f"{s} 已停止"})
            except Exception as e:
                results.append({"service": s, "success": False, "error": str(e)})
        else:
            results.append({"service": s, "success": False, "error": f"{s} 不存在"})

    return {"results": results}

def restart_service(service_name: str):
    """
    使用 docker SDK 重啟 container
    """
    if not service_name:
        return {"success": False, "error": "missing service name"}

    try:
        # 讀 compose_state.json
        with open(COMPOSE_STATE_PATH, "r", encoding="utf-8") as f:
            compose_state = json.load(f)

        if service_name not in compose_state or not compose_state[service_name].get("exists"):
            return {"success": False, "error": f"{service_name} 不存在或未啟用"}

        try:
            service_name = service_name.lower()
            container = client.containers.get(service_name)
            container.restart()
        except docker.errors.NotFound:
            return {"success": False, "error": f"{service_name} container 不存在"}
        except Exception as e:
            return {"success": False, "error": f"重啟失敗: {str(e)}"}

        return {"success": True, "message": f"{service_name} 服務已重啟"}

    except Exception as e:
        return {"success": False, "error": str(e)}

def update_compose_state(file_path, service, model=None, exists=None, running=None):
    # 讀取 JSON
    with open(file_path, "r", encoding="utf-8") as f:
        compose_state = json.load(f)

    if service not in compose_state:
        return {"success": False, "error": f"{service} 不存在"}

    # 更新欄位
    if model is not None:
        compose_state[service]["model"] = model

    if exists is not None:
        compose_state[service]["exists"] = exists

    if running is not None:
        compose_state[service]["running"] = running

    # 寫回 JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(compose_state, f, indent=2, ensure_ascii=False)

    return {"success": True}

def sync_container_status(compose_state_path):
    with open(compose_state_path, "r", encoding="utf-8") as f:
        compose_state = json.load(f)

    result = {}

    # 先收集要啟動和要停止的服務
    to_start = []
    to_stop = []

    for service, state in compose_state.items():
        cmd = f"docker ps --filter name={service.lower()} --format '{{{{.Names}}}}'"
        proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()
        running = bool(out.strip())
        exists = state.get("exists", False)

        if running and not exists:
            to_stop.append(service)
        elif not running and exists:
            to_start.append(service)

        # 先初始化 result
        result[service] = {
            "exists": exists,
            "running": running
        }

    # 一次關掉所有不應該運行的服務
    if to_stop:
        stop_containers(" ".join([s.lower() for s in to_stop]))
        for s in to_stop:
            compose_state[s]["running"] = False
            result[s]["running"] = False

    # 一次啟動所有應該運行的服務
    if to_start:
        start_containers(" ".join([s.lower() for s in to_start]))
        for s in to_start:
            compose_state[s]["running"] = True
            result[s]["running"] = True

    # 寫回 JSON
    with open(compose_state_path, "w", encoding="utf-8") as f:
        json.dump(compose_state, f, indent=2, ensure_ascii=False)

    return {"success": True, "data": result}

def get_gpu_info():
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits",
            shell=True,
            capture_output=True,
            text=True
        )

        output = result.stdout.strip()

        if not output:
            return {"success": False, "error": "nvidia-smi returned empty output"}

        line = output.split("\n")[0]
        parts = [x.strip() for x in line.split(",")]

        if len(parts) != 3:
            return {
                "success": False,
                "error": "unexpected output format",
                "raw": output
            }

        usage, mem_used, mem_total = parts

        return {
            "success": True,
            "usage": int(usage),
            "memory_used": int(mem_used),
            "memory_total": int(mem_total)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    
def bytes_to_gb(x_mb):
    """將 MB 轉成 GB，保留一位小數"""
    return round(x_mb / 1024, 1)


def get_container_gpu_stats(compose_state_path):
    # 讀 compose state
    with open(compose_state_path, "r", encoding="utf-8") as f:
        compose_state = json.load(f)

    services = [s for s in compose_state if compose_state[s].get("exists")]

    result = {s: {"used_mb": 0} for s in services}

    # 取得 GPU total memory (MB)
    gpu_total_cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"

    gpu_total_proc = subprocess.run(
        gpu_total_cmd, shell=True, capture_output=True, text=True
    )

    gpu_total_mb = int(gpu_total_proc.stdout.strip().split("\n")[0])

    # 取得 GPU process
    cmd = (
        "nvidia-smi --query-compute-apps=pid,used_memory "
        "--format=csv,noheader,nounits"
    )

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if proc.returncode != 0:
        return {"success": False, "error": proc.stderr}

    lines = proc.stdout.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue

        pid_str, mem_str = [x.strip() for x in line.split(",")]
        pid = int(pid_str)
        mem_mb = int(mem_str)

        try:
            containers = client.containers.list(filters={"pid": pid})

            if not containers:
                continue

            name = containers[0].name

            if name in result:
                result[name]["used_mb"] += mem_mb

        except Exception:
            continue

    # 格式化輸出
    formatted = {}
    for service in result:
        used_gb = bytes_to_gb(result[service]["used_mb"])
        total_gb = bytes_to_gb(gpu_total_mb)
        percent = round((result[service]["used_mb"] / gpu_total_mb) * 100, 2) if gpu_total_mb else 0

        formatted[service] = f"{used_gb}GB / {total_gb}GB ({percent}%)"

    return {
        "success": True,
        "data": formatted
    }

def get_models_func(file_path, service: str):
    """取得 service 的 model list 與使用中模型"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            compose_state = json.load(f)
        models = compose_state.get(service, {}).get("model", {}).get("list", [])
        use_model = compose_state.get(service, {}).get("model", {}).get("use")
        return {"success": True, "list": models, "use": use_model}
    except Exception as e:
        return {"success": False, "error": str(e)}

def download_model_func(file_path, service: str, model_name: str):
    """將 model 加入 list，設定為使用中，並下載到 volume（使用 huggingface_hub API）"""
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
        
        # 下載整個模型 repo（snapshot_download 會自動抓最新 commit）
        download_model_three_method(service, model_name, service_folder)

        return {"success": True, "message": f"{model_name} 已加入列表並下載完成", "list": model_list}

    except Exception as e:
        return {"success": False, "error": str(e)}

def download_model_three_method(service,name,dir):
    try:
        if service in ["Chat", "Embedding", "STT"]:
            local_dir = os.path.join(dir, 'hub')
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
    except:
        return

def delete_model_func(file_path, service: str, model_name: str):
    """刪除 model，使用中模型無法刪除，並移除 volume 內對應資料夾"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            compose_state = json.load(f)

        service_info = compose_state.get(service, {})
        model_info = service_info.get("model", {})
        model_list = model_info.get("list", [])

        # 1️⃣ 檢查是否正在使用中
        if model_info.get("use") == model_name:
            return {"success": False, "message": f"{model_name} 目前正在使用中，無法刪除", "list": model_list}

        # 2️⃣ 刪除 list 中的模型
        if model_name in model_list:
            model_list.remove(model_name)
            compose_state[service]["model"]["list"] = model_list  # 更新 list

        # 3️⃣ 刪除 Volume 中的模型資料夾
        base_path = "/app/models"
        service_folder = os.path.join(base_path, service.lower())

        repo_folder = "models--" + model_name.replace("/", "--")
        model_path = os.path.join(service_folder, repo_folder)

        # 刪模型
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # 刪 lock
        lock_folder = os.path.join(service_folder, ".locks")
        lock_file = os.path.join(lock_folder, repo_folder + ".lock")

        if os.path.exists(lock_file):
            os.remove(lock_file)

        # 4️⃣ 寫回 compose_state
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(compose_state, f, indent=2)

        return {"success": True, "message": f"{model_name} 已刪除（含 volume 資料）", "list": model_list}

    except Exception as e:
        return {"success": False, "error": str(e)}

def select_model_func(file_path, service: str, model_name: str):
    """
    將指定模型設為使用中（不下載，只更新 use）
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            compose_state = json.load(f)

        service_info = compose_state.get(service)
        if not service_info:
            return {"success": False, "error": f"服務 {service} 不存在"}

        model_list = service_info.get("model", {}).get("list", [])
        if model_name not in model_list:
            return {"success": False, "error": f"{model_name} 不在模型列表中"}

        # 設為使用中
        compose_state[service]["model"]["use"] = model_name

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(compose_state, f, indent=2)

        set_service_model_func(service, model_name)

        return {"success": True, "message": f"{model_name} 已設為使用中", "use": model_name}

    except Exception as e:
        return {"success": False, "error": str(e)}
    
ENV_FILE = "/app/.env"

MODEL_ENV_MAP = {
    "Chat": "CHAT_MODEL",
    "Embedding": "EMBEDDING_MODEL",
    "TTS": "TTS_MODEL",
    "STT": "STT_MODEL"
}

def set_service_model_func(service: str, model_path: str):
    try:
        service = service

        if service not in MODEL_ENV_MAP:
            return {"success": False, "error": f"未知 service: {service}"}

        env_key = MODEL_ENV_MAP[service]

        if not os.path.exists(ENV_FILE):
            return {"success": False, "error": ".env 不存在"}

        with open(ENV_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        found = False

        for line in lines:
            if line.startswith(env_key + "="):
                new_lines.append(f"{env_key}={model_path}\n")
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f"{env_key}={model_path}\n")

        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return {
            "success": True,
            "message": f"{service} 模型已設定為 {model_path}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}














# OPENAI API (其他服務) 
import uuid
import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import StreamingHttpResponse, JsonResponse , HttpResponse
from rest_framework import status

# Docker 內部服務對應 API
SERVICE_MAP = {
    "License" : "http://license:8000",
    "Chat": "http://chat:8000",
    "Embedding": "http://embedding:8000",
    "TTS": "http://tts:8000",
    "STT": "http://stt:8000",
}

def check_license(required_features: list[str]):
    """
    檢查 license 是否有效，以及是否包含需要的 features
    """
    def decorator(func):
        def wrapper(self, request, *args, **kwargs):
            try:
                r = requests.get(f"{SERVICE_MAP['License']}/verify", timeout=5)
                r.raise_for_status()
                data = r.json()

                if not data.get("valid"):
                    return Response({"error": "License invalid"}, status=403)

                features = data.get("features", [])
                # 檢查所需 feature
                missing = [f for f in required_features if f not in features]
                if missing:
                    return Response(
                        {"error": f"Missing licensed feature(s): {missing}"},
                        status=403
                    )

            except requests.exceptions.RequestException:
                return Response({"error": "Cannot reach license server"}, status=500)
            except Exception as e:
                return Response({"error": f"License check failed: {str(e)}"}, status=403)

            return func(self, request, *args, **kwargs)

        return wrapper
    return decorator

def generate_license_request(private_key_file, user, days=30, features='[]'):
    """
    呼叫 LICENSE API 生成 license.key 並上傳
    """
    files = {"private_key_file": private_key_file}
    data = {"user": user, "days": days, "features": features}

    # 生成 license
    r = requests.post(f"{SERVICE_MAP['License']}/generate", files=files, data=data)
    r.raise_for_status()

    # 上傳 license
    files_upload = {"file": ("license.key", r.content)}
    r2 = requests.post(f"{SERVICE_MAP['License']}/upload-license", files=files_upload)
    r2.raise_for_status()

    return r2.json()


def upload_license_request(license_file):
    """
    呼叫 LICENSE API 上傳 license.key
    """
    files = {"file": license_file}
    r = requests.post(f"{SERVICE_MAP['License']}/upload-license", files=files)
    r.raise_for_status()
    return r.json()

def filter_none(d):
    """過濾掉值為 None 的參數，避免下游收到預設值"""
    return {k: v for k, v in d.items() if v is not None}

class ChatCompletions(APIView):
    """POST /v1/chat/completions"""
    @check_license(["Chat"])
    def post(self, request):
        payload = request.data
        stream = payload.get("stream", False)

        try:
            r = requests.post(
                f"{SERVICE_MAP['Chat']}/v1/chat/completions",
                json=payload,
                stream=stream,
                timeout=300
            )
            if stream:
                def event_stream():
                    for line in r.iter_lines():
                        if line:
                            yield line.decode("utf-8") + "\n"

                return StreamingHttpResponse(
                    event_stream(),
                    content_type="text/event-stream"
                )
            return Response(r.json(), status=r.status_code)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)

class Embeddings(APIView):
    """POST /v1/embeddings"""
    @check_license(["Embedding"])
    def post(self, request):
        payload = request.data
        try:
            r = requests.post(
                f"{SERVICE_MAP['Embedding']}/v1/embeddings",
                json=payload,
                timeout=120
            )
            return Response(r.json(), status=r.status_code)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)
        
class Embed(APIView):
    """POST /embed"""
    @check_license(["Embedding"])
    def post(self, request):
        try:
            r = requests.post(
                f"{SERVICE_MAP['Embedding']}/embed",
                json=request.data,
                timeout=120
            )
            r.raise_for_status()
            return Response(r.json(), status=r.status_code)

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)

class Transcriptions(APIView):
    """POST /v1/audio/transcriptions"""
    @check_license(["STT"])
    def post(self, request):
        try:
            audio = request.FILES.get("file")
            files = None
            if audio:
                files = {
                    "file": (audio.name, audio, audio.content_type)
                }
            data = {
                k: v for k, v in request.data.items()
                if k != "file"
            }
            r = requests.post(
                f"{SERVICE_MAP['STT']}/v1/audio/transcriptions",
                data=data,
                files=files,
                timeout=120
            )
            return Response(r.json(), status=r.status_code)
        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)

class CreateSpeaker(APIView):
    """POST /v1/speakers"""
    @check_license(["TTS"])
    def post(self, request):
        try:
            r = requests.post(
                f"{SERVICE_MAP['TTS']}/v1/speakers",
                json=request.data,
                timeout=60
            )
            return Response(r.json(), status=r.status_code)
        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)

class SpeechSynthesis(APIView):
    """POST /v1/audio/speech"""

    @check_license(["TTS"])
    def post(self, request):

        payload = request.data
        stream = payload.get("stream", False)

        try:
            r = requests.post(
                f"{SERVICE_MAP['TTS']}/v1/audio/speech",
                json=payload,
                stream=stream,
                timeout=120
            )

            if stream:
                return StreamingHttpResponse(
                    r.iter_content(chunk_size=4096),
                    content_type=r.headers.get("Content-Type", "audio/mpeg")
                )

            resp = HttpResponse(
                r.content,
                content_type=r.headers.get("Content-Type", "audio/mpeg")
            )

            resp["Content-Disposition"] = 'inline; filename="output.mp3"'
            return resp

        except requests.exceptions.RequestException as e:
            return Response({"error": str(e)}, status=500)