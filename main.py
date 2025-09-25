# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io, os, numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI(title="Clothes Matching TFLite API (Docker)")

# Allow CORS for testing/demo. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/model.tflite"
INPUT_SIZE = 224
LABELS = ["Jeans", "LongSleevedTop", "Shorts", "Skirt", "Tee"]  # change if needed

interpreter = None
input_details = None
output_details = None

def load_tflite_model():
    global interpreter, input_details, output_details
    if interpreter is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def preprocess_image(file_bytes: bytes, size=INPUT_SIZE):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    w, h = img.size
    min_side = min(w,h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0).astype(np.float32)
    return inp, img

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

def dominant_color(pil_img: Image.Image):
    small = pil_img.resize((40, 40))
    arr = np.array(small).reshape(-1, 3)
    avg = np.mean(arr, axis=0).astype(int)
    return tuple(avg.tolist())

@app.on_event("startup")
def startup_event():
    try:
        load_tflite_model()
        print("TFLite model loaded.")
    except Exception as e:
        print("Model load failed:", e)

@app.get("/")
def root():
    return {"status":"ok", "msg":"Clothes Matching TFLite API"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    results = []
    for f in files:
        content = await f.read()
        try:
            inp, pil_img = preprocess_image(content, size=INPUT_SIZE)
            in_dtype = input_details[0]['dtype']
            if in_dtype == np.uint8:
                inp_to_set = (inp * 255).astype(np.uint8)
            else:
                inp_to_set = inp.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], inp_to_set)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index']).squeeze()
            probs = softmax(out) if out.ndim == 1 else out
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            label = LABELS[top_idx] if top_idx < len(LABELS) else str(top_idx)
            main_rgb = dominant_color(pil_img)
            results.append({
                "filename": f.filename,
                "label": label,
                "confidence": round(confidence, 4),
                "main_color": "rgb({},{},{})".format(*main_rgb)
            })
        except Exception as e:
            results.append({
                "filename": getattr(f, "filename", "unknown"),
                "error": str(e)
            })
    return {"predictions": results}

# 🔹 New endpoint for frontend (returns simplified list)
@app.post("/classify")
async def classify(files: List[UploadFile] = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    response = []
    for f in files:
        content = await f.read()
        try:
            inp, pil_img = preprocess_image(content, size=INPUT_SIZE)
            in_dtype = input_details[0]['dtype']
            if in_dtype == np.uint8:
                inp_to_set = (inp * 255).astype(np.uint8)
            else:
                inp_to_set = inp.astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], inp_to_set)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index']).squeeze()
            probs = softmax(out) if out.ndim == 1 else out
            top_idx = int(np.argmax(probs))
            label = LABELS[top_idx] if top_idx < len(LABELS) else str(top_idx)
            main_rgb = dominant_color(pil_img)
            response.append({
                "category": label,
                "dominant_color": "rgb({},{},{})".format(*main_rgb)
            })
        except Exception as e:
            response.append({
                "category": "error",
                "dominant_color": "rgb(0,0,0)",
                "error": str(e)
            })
    return response

