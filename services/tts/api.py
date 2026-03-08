import os, sys
from pathlib import Path
import argparse
import shutil
import logging
from logging.handlers import RotatingFileHandler
import datetime
import random
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.cli.cosyvoice import AutoModel

import torch
import io
from pydub import AudioSegment
from flask import Flask, request, jsonify, Response, stream_with_context
from opencc import OpenCC
import uuid

os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"
os.environ["TORCH_HOME"] = "/root/.cache/huggingface"


cc = OpenCC('t2s') 

# --- Flask App Initialization ---
app = Flask(__name__)

def openai_error(message, type_, param=None, code=None, status=400):
    response = {
        "error": {
            "message": message,
            "type": type_,
            "param": param,
            "code": code
        }
    }
    return jsonify(response), status
# --- Logging Setup ---
def setup_logging(logs_dir: Path):
    log = logging.getLogger('werkzeug')
    log.handlers[:] = []
    log.setLevel(logging.WARNING)

    root_log = logging.getLogger()
    root_log.handlers = []
    root_log.setLevel(logging.WARNING)

    app.logger.setLevel(logging.WARNING)
    log_file = logs_dir / f'{datetime.datetime.now().strftime("%Y%m%d")}.log'
    file_handler = RotatingFileHandler(str(log_file), maxBytes=1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

# --- Core Functions ---

def setup_environment():
    """Sets up PYTHONPATH for Matcha-TTS and validates ffmpeg availability."""
    root_dir = Path(__file__).parent
    matcha_tts_path = root_dir / 'third_party' / 'Matcha-TTS'
    if str(matcha_tts_path) not in sys.path:
        sys.path.append(str(matcha_tts_path))

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH. Please ensure it is installed and accessible.")
        # Simple check for homebrew path on macOS
        if sys.platform == 'darwin' and (Path("/opt/homebrew/bin") / "ffmpeg").exists():
             os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ["PATH"]

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg could not be found. Please install it and add it to your system's PATH.")
    print(f"ffmpeg found at: {shutil.which('ffmpeg')}")

def tensor_to_mp3_bytes(tensor, sample_rate):
    if tensor.dtype != torch.int16:
        audio_np = (tensor.squeeze().cpu().numpy() * 32767).astype('int16')
    else:
        audio_np = tensor.squeeze().cpu().numpy()

    channels = 1 if audio_np.ndim == 1 else audio_np.shape[0]
    audio_seg = AudioSegment(
        audio_np.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=channels
    )

    mp3_io = io.BytesIO()
    audio_seg.export(mp3_io, format="mp3", bitrate="128k")
    return mp3_io.getvalue()

def batch_stream(params, args):
    seed = args.seed
    api_seed = params.get('seed', -1)
    if api_seed != -1:
        seed = api_seed
    if seed == -1:
        seed = random.randint(1, 100000000)
    set_all_random_seed(seed)

    inference_stream = model.inference_zero_shot(params['text'],'', '', zero_shot_spk_id=params['role'], speed=params['speed'], stream=True)

    for output in inference_stream:
        tts_speech = output['tts_speech']
        mp3_chunk = tensor_to_mp3_bytes(tts_speech, model.sample_rate)
        yield mp3_chunk

# --- Flask Routes ---

@app.route('/v1/audio/speech', methods=['POST'])
def audio_speech():
    request_id = str(uuid.uuid4())

    if not request.is_json:
        return openai_error(
            message="Request must be JSON",
            type_="invalid_request_error",
            code="invalid_json",
            status=400
        )

    data = request.get_json()

    if 'input' not in data:
        return openai_error(
            message="'input' is required",
            type_="invalid_request_error",
            param="input",
            code="missing_parameter",
            status=422
        )

    if 'voice' not in data:
        return openai_error(
            message="'voice' is required",
            type_="invalid_request_error",
            param="voice",
            code="missing_parameter",
            status=422
        )

    params = {
        'text': data['input'],
        'speed': float(data.get('speed', 1.0)),
        'role': data['voice'],
        'reference_audio': None
    }

    params['text'] = cc.convert(params['text'])

    VOICE_LIST = model.list_available_spks()
    print(VOICE_LIST)
    if params['role'] not in VOICE_LIST and not os.path.exists(params['role']):
        return openai_error(
            message=f"Voice '{params['role']}' not found",
            type_="invalid_request_error",
            param="voice",
            code="voice_not_found",
            status=404
        )

    def generate():
        try:
            yield from batch_stream(params, app.config['args'])
        except Exception as e:
            app.logger.error(f"[{request_id}] {e}", exc_info=True)
            return

    return Response(
        stream_with_context(generate()),
        mimetype='audio/mpeg',
        headers={
            "Content-Disposition": "attachment; filename=speech.mp3",
            "X-Request-ID": request_id
        }
    )


@app.route('/v1/speakers', methods=['POST'])
def make_spks():
    """
    POST JSON 範例:
    {
        "speakers": [
            {
                "text": "希望你以后能够做的比我还好呦。",
                "audio_path": "./asset/zero_shot_prompt1.wav",
                "spk_name": "my_zero_shot_spk1"
            },
            {
                "text": "希望你以后能够做的比我还好呦。",
                "audio_path": "./asset/zero_shot_prompt2.wav",
                "spk_name": "my_zero_shot_spk2"
            }
        ]
    }
    """
    if not request.is_json:
        return openai_error(
            message="Request must be JSON",
            type_="invalid_request_error",
            code="invalid_json",
            status=400
        )

    data = request.json
    speakers = data.get('speakers')

    if not isinstance(speakers, list):
        return openai_error(
            message="'speakers' must be a list",
            type_="invalid_request_error",
            param="speakers",
            code="invalid_type",
            status=422
        )

    added_spks = []
    errors = []

    for spk in speakers:
        text = spk.get('text')
        audio_path = spk.get('audio_path')
        spk_name = spk.get('spk_name')

        if not text or not audio_path or not spk_name:
            errors.append({
                "spk_name": spk_name,
                "error": {
                    "message": "Missing text, audio_path, or spk_name",
                    "type": "invalid_request_error",
                    "code": "missing_parameter"
                }
            })
            continue

        if not os.path.exists(audio_path):
            errors.append({
                "spk_name": spk_name,
                "error": {
                    "message": f"File {audio_path} not found",
                    "type": "invalid_request_error",
                    "code": "file_not_found"
                }
            })
            continue

        try:
            model.add_zero_shot_spk(text, audio_path, spk_name)
            added_spks.append(spk_name)
        except Exception as e:
            errors.append({
                "spk_name": spk_name,
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": "model_error"
                }
            })

    if added_spks:
        model.save_spkinfo()

    return jsonify({
        "object": "speaker.batch",
        "added": added_spks,
        "errors": errors
    }), 200


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice API Server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to.')
    parser.add_argument('--models-dir', type=str, default='./pretrained_models', help='Directory to store and load models from.')
    parser.add_argument('--output-dir', type=str, default='./tmp', help='Directory to save generated audio files.')
    parser.add_argument('--refer-audio-dir', type=str, default='.', dest='refer_audio_dir', help='Base directory for reference audio files.')
    parser.add_argument('--seed', type=int, default=-1, help='Global random seed. -1 for random. Overridden by seed in API call.')
    parser.add_argument('--preload-models', type=str, default='CosyVoice2-0.5B', help='Model directories to preload')
    parser.add_argument('--disable-download', action='store_true', help='Disable automatic model downloading.')
    args = parser.parse_args()

    app.config['args'] = args

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    app.static_folder = str(output_dir)
    app.static_url_path = '/' + output_dir.name

    setup_logging(logs_dir)
    setup_environment()
    global model
    if args.preload_models:
        model_dir = Path(args.models_dir) / args.preload_models
        try:
            model = AutoModel(model_dir=model_dir)
        except Exception as e:
            app.logger.error(f"Failed to preload model '{model_dir}': {e}", exc_info=True)
            sys.exit(1)

    print(f"\n--- CosyVoice API Server ---")
    print(f"- Host: {args.host}")
    print(f"- Port: {args.port}")
    print(f"- Models Dir: {Path(args.models_dir).resolve()}")
    print(f"- Output Dir: {Path(args.output_dir).resolve()}")
    print(f"- Reference Dir: {Path(args.refer_audio_dir).resolve()}")
    print(f"- Preloaded models: {args.preload_models if args.preload_models else 'None'}")
    print(f"- Auto-download: {'Disabled' if args.disable_download else 'Enabled'}")
    print(f"- API running at: http://{args.host}:{args.port}")
    print(f"----------------------------")

    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port)
    except ImportError:
        app.run(host=args.host, port=args.port)
