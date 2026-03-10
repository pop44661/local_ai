import base64
import json
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

app = FastAPI()

with open("public.pem", "rb") as f:
    public_key = serialization.load_pem_public_key(f.read())

LICENSE_PATH = "/license.key"

def load_license():
    try:
        with open(LICENSE_PATH) as f:
            return json.load(f)
    except:
        raise HTTPException(404, "license.key not found")


def verify_license():
    lic = load_license()

    payload = base64.b64decode(lic["payload"])
    signature = base64.b64decode(lic["signature"])

    public_key.verify(
        signature,
        payload,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    data = json.loads(payload)

    now = int(time.time())

    if now < data["start"]:
        raise Exception("license not active")

    if now > data["expire"]:
        raise Exception("license expired")

    return data

@app.post("/upload-license")
async def upload_license(file: UploadFile = File(...)):

    content = await file.read()

    with open(LICENSE_PATH, "wb") as f:
        f.write(content)

    return {"message": "license uploaded"}

@app.post("/generate")
async def generate_license(
    private_key_file: UploadFile = File(...),
    user: str = Form(...),
    days: int = Form(30),
    start: int | None = Form(None),
    features: str = Form("[]")
):

    # 讀 private key
    private_key_data = await private_key_file.read()

    private_key = serialization.load_pem_private_key(
        private_key_data,
        password=None
    )

    if start is None:
        start = int(time.time())

    expire = start + days * 86400

    features_list = json.loads(features)

    payload = {
        "user": user,
        "start": start,
        "expire": expire,
        "features": features_list
    }

    payload_bytes = json.dumps(payload).encode()

    signature = private_key.sign(
        payload_bytes,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    lic = {
        "payload": base64.b64encode(payload_bytes).decode(),
        "signature": base64.b64encode(signature).decode()
    }

    path = "/tmp/license.key"

    with open(path, "w") as f:
        json.dump(lic, f)

    return FileResponse(path, filename="license.key")


@app.get("/verify")
def verify():

    try:
        data = verify_license()

        return {
            "valid": True,
            "user": data["user"],
            "start": data["start"],
            "expire": data["expire"],
            "features": data["features"]
        }

    except Exception as e:
        raise HTTPException(403, f"license invalid: {str(e)}")