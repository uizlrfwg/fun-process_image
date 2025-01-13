import os
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import numpy as np
import onnxruntime as ort
import torchvision.transforms as transforms
from pydantic import BaseModel
from typing import List, Dict, Any, Set
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

API_KEYS: Set[str] = {
    "abc123",
    "abc123-1",
    "abc123-2",
}

MODEL_PATH = "./mod/3d_rollball_objects.onnx"
MODEL_URL = "https://api.onedrive.com/v1.0/shares/s!Ag2sFwUQFgNLbA8M8JAfNDFjDJA/root/content"

transform_3d = transforms.Compose([
    transforms.Resize((52, 52)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

class PredictionResponse(BaseModel):
    predicted_index: int
    max_similarity: str
    all_similarities: List[str]

class TaskRequest(BaseModel):
    clientKey: str
    task: Dict[str, Any]

model = None

def download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading model...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Model downloaded successfully.")
    else:
        raise RuntimeError("Failed to download the model. Please check the URL.")

def init_model():
    if not os.path.exists(MODEL_PATH):
        download_model()

    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
        sess_options=ort.SessionOptions()
    )

def format_probability(prob: float) -> str:
    return f"{prob:.14f}"

def verify_api_key(client_key: str) -> bool:
    return client_key in API_KEYS

async def predict_3d(image: Image.Image, model: ort.InferenceSession):
    img = image.convert('RGB')
    left_image = img.crop((0, 200, 200, 400))
    left_image = transform_3d(left_image).numpy()
    left_image = np.expand_dims(left_image, axis=0)

    width, _ = img.size
    total_right_images = width // 200
    img_rights = [transform_3d(img.crop((200 * j, 0, 200 * (j + 1), 200))).numpy()
                  for j in range(total_right_images)]

    max_similarity = -1
    most_similar_index = -1
    all_similarities = []

    for i, img_right in enumerate(img_rights):
        img_right = np.expand_dims(img_right, axis=0)
        outputs = await run_in_threadpool(
            lambda: model.run(None, {
                "input_left": left_image,
                "input_right": img_right
            })
        )
        distance = outputs[0][0][0]
        all_similarities.append(distance)
        if distance > max_similarity:
            max_similarity = distance
            most_similar_index = i

    return most_similar_index, max_similarity, all_similarities

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = init_model()
    yield
    model = None

app = FastAPI(title="3D Model Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process_image", response_model=Dict[str, Any])
async def process_image(request: TaskRequest):
    try:
        if not verify_api_key(request.clientKey):
            return JSONResponse(
                status_code=403,
                content={"errorCode": "INVALID_API_KEY", "status": "error", "taskId": request.task.get('taskId', 'N/A')}
            )

        if request.task['type'] != "FunCaptcha" or request.task.get('question') != "3d_rollball_objects":
            return JSONResponse(
                status_code=400,
                content={"errorCode": "INVALID_QUESTION_TYPE", "status": "error", "taskId": request.task.get('taskId', 'N/A')}
            )

        try:
            image_data = base64.b64decode(request.task['image'])
            img = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"errorCode": "INVALID_IMAGE_DATA", "status": "error", "taskId": request.task.get('taskId', 'N/A')}
            )

        predicted_index, max_similarity, all_similarities = await predict_3d(img, model)

        formatted_similarities = [format_probability(p) for p in all_similarities]
        print(f"Predicted index: {predicted_index}, Max similarity: {max_similarity}, All similarities: {formatted_similarities}")

        return {
            "errorCode": "",
            "solution": {
                "confidences": formatted_similarities,
                "objects": [predicted_index],
                "predicted_index": int(predicted_index),
                "label": f"object_{predicted_index}"
            },
            "taskId": request.task.get('taskId', 'N/A'),
            "status": "ready",
            "errorId": 0
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"errorCode": "SERVER_ERROR", "status": "error", "taskId": request.task.get('taskId', 'N/A')}
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=6789,
        workers=1,
        loop="uvloop",
        limit_concurrency=1000,
        timeout_keep_alive=30,
    )
