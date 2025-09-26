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
LABELS = ["Jeans", "LongSleevedTop", "Shorts", "Skirt", "Tee"]  # 5 categories
TEMP_DIR = "temp_images"

os.makedirs(TEMP_DIR, exist_ok=True)  # ensure temp folder exists

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
    print("✅ TFLite model loaded successfully.")
    print("Input details:", input_details)
    print("Output details:", output_details)

def preprocess_image(file_bytes: bytes, filename: str, size=224):
    """Crop to square, resize to model input, normalize to float32 0-1, and save resized image."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Crop to square
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))

    # Resize to 224x224
    img = img.resize((size, size), Image.BILINEAR)

    # Save resized image temporarily
    save_path = os.path.join(TEMP_DIR, f"{os.path.splitext(filename)[0]}_resized.png")
    img.save(save_path)

    # Convert to numpy array and normalize
    arr = np.array(img).astype(np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0)

    print(f"[PREPROCESS] {filename} -> shape={inp.shape}, dtype={inp.dtype}, saved={save_path}")
    return inp, img, save_path


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
    except Exception as e:
        print("❌ Model load failed:", e)

@app.get("/")
def root():
    return {"status": "ok", "msg": "Clothes Matching TFLite API"}

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    results = []
    for f in files:
        content = await f.read()
        try:
            inp, pil_img, save_path = preprocess_image(content, f.filename, size=INPUT_SIZE)
            
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            
            out = interpreter.get_tensor(output_details[0]['index'])
            out = out.squeeze()
            if out.ndim == 0:
                out = np.array([out])
            
            if out.max() > 1.0 or out.min() < 0.0:
                probs = softmax(out)
            else:
                probs = out

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            label = LABELS[top_idx]
            main_rgb = dominant_color(pil_img)
            results.append({
                "filename": f.filename,
                "resized_path": save_path,
                "resized_size": f"{pil_img.size[0]}x{pil_img.size[1]}",
                "mode": pil_img.mode,
                "label": label,
                "confidence": round(confidence, 4),
                "main_color": f"rgb({main_rgb[0]},{main_rgb[1]},{main_rgb[2]})"
            })
        except Exception as e:
            results.append({
                "filename": getattr(f, "filename", "unknown"),
                "error": str(e)
            })
    return {"predictions": results}

@app.post("/classify")
async def classify(files: List[UploadFile] = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    response = []
    for f in files:
        content = await f.read()
        try:
            inp, pil_img, save_path = preprocess_image(content, f.filename, size=INPUT_SIZE)
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index']).squeeze()
            probs = softmax(out) if out.ndim == 1 else out
            top_idx = int(np.argmax(probs))
            label = LABELS[top_idx] if top_idx < len(LABELS) else str(top_idx)
            main_rgb = dominant_color(pil_img)
            print(f"[CLASSIFY] {f.filename} -> {label}, probs={probs}")
            response.append({
                "filename": f.filename,
                "resized_path": save_path,
                "resized_size": f"{pil_img.size[0]}x{pil_img.size[1]}",
                "mode": pil_img.mode,
                "category": label,
                "dominant_color": f"rgb({main_rgb[0]},{main_rgb[1]},{main_rgb[2]})"
            })
        except Exception as e:
            response.append({
                "filename": getattr(f, "filename", "unknown"),
                "error": str(e),
                "category": "error",
                "dominant_color": "rgb(0,0,0)"
            })
    return response


