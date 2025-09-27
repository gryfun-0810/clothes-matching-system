# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import io, os, numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import cv2
import math

app = FastAPI(title="Clothes Matching TFLite API (Docker)")

# Allow CORS for testing/demo. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "" # "model/model.tflite"
INPUT_SIZE = 224
LABELS = ["Jeans", "LongSleevedTop", "Shorts", "Skirt", "Tee"]  # 5 categories
TEMP_DIR = "temp_images"

os.makedirs(TEMP_DIR, exist_ok=True)  # ensure temp folder exists

interpreter = None
input_details = None
output_details = None

# New: Pydantic models for color matching
class ClothingItem(BaseModel):
    filename: str
    category: str
    dominant_color: str
    type: str

class ColorMatchRequest(BaseModel):
    selected_item: ClothingItem
    all_items: List[ClothingItem]

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

# ---------- New: Color similarity functions ----------
def parse_rgb_color(rgb_string):
    """Parse 'rgb(r,g,b)' string to (r,g,b) tuple"""
    try:
        # Remove 'rgb(' and ')' and split by comma
        rgb_values = rgb_string.replace('rgb(', '').replace(')', '').split(',')
        return tuple(int(val.strip()) for val in rgb_values)
    except:
        return (128, 128, 128)  # Default gray if parsing fails

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV"""
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Hue calculation
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif max_val == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation calculation
    s = 0 if max_val == 0 else (diff / max_val)
    
    # Value calculation
    v = max_val
    
    return h, s, v

def calculate_color_similarity(color1_rgb, color2_rgb):
    """Calculate color similarity between two RGB colors using HSV distance"""
    try:
        rgb1 = parse_rgb_color(color1_rgb)
        rgb2 = parse_rgb_color(color2_rgb)
        
        # Convert to HSV for better color comparison
        h1, s1, v1 = rgb_to_hsv(*rgb1)
        h2, s2, v2 = rgb_to_hsv(*rgb2)
        
        # Calculate hue difference (circular distance)
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180.0  # Normalize to 0-1
        
        # Calculate saturation and value differences
        sat_diff = abs(s1 - s2)
        val_diff = abs(v1 - v2)
        
        # Weighted similarity score (hue is most important for color matching)
        similarity = 1.0 - (0.6 * hue_diff + 0.2 * sat_diff + 0.2 * val_diff)
        
        # Apply color matching rules
        # Neutral colors (low saturation) match well with anything
        if s1 < 0.3 or s2 < 0.3:
            similarity += 0.2
        
        # Complementary colors (opposite hues) can work well
        if 150 <= abs(h1 - h2) <= 210:
            similarity += 0.1
            
        return max(0.0, min(1.0, similarity))  # Clamp to 0-1
        
    except Exception as e:
        print(f"Color similarity calculation error: {e}")
        return 0.5  # Default moderate similarity

def get_opposite_type(clothing_type):
    """Get the opposite clothing type for matching"""
    if clothing_type == "top":
        return "bottom"
    elif clothing_type == "bottom":
        return "top"
    else:
        return None
# ---------- end new color functions ----------

# ---------- New: HSV + k-means color extraction ----------
def pil_to_cv2_rgb(pil_img: Image.Image):
    """Convert PIL RGB image to OpenCV RGB numpy array"""
    return np.array(pil_img)  # PIL uses RGB, OpenCV uses BGR but we'll convert appropriately when needed

def extract_top_k_colors(pil_img: Image.Image, k=3, attempts=5):
    """
    Return top-k colors as RGB tuples (r,g,b) using HSV+kmeans.
    Steps:
      - Convert PIL RGB -> OpenCV HSV
      - k-means on HSV pixels (so clustering happens in HSV space)
      - convert cluster centers (HSV) back to RGB
      - return list of RGB tuples ordered by cluster size (largest first)
    """
    # Convert PIL->numpy RGB
    rgb = pil_to_cv2_rgb(pil_img)  # shape (H,W,3), RGB
    # Convert RGB -> BGR (OpenCV default) then to HSV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Prepare data for kmeans: use HSV channels, reshape to (N,3) and convert to float32
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Filter out almost-transparent/black pixels? (optional) - here we keep all pixels
    # Run k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS
    if len(pixels) < k:
        # fallback: if image is tiny or single-color
        centers = np.unique(pixels, axis=0)
        centers = centers[:k].astype(np.uint8)
        labels = np.zeros(len(pixels), dtype=np.int32)
    else:
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, flags)
        centers = centers.astype(np.uint8)  # HSV centers

    # Count cluster sizes to order them
    label_counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-label_counts)  # descending

    rgb_centers = []
    for idx in order:
        hsv_center = centers[idx].reshape(1,1,3)  # shape (1,1,3) for cvt
        # Convert HSV center (uint8) -> BGR -> RGB
        bgr_center = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)
        rgb_center = cv2.cvtColor(bgr_center, cv2.COLOR_BGR2RGB)
        r, g, b = int(rgb_center[0,0,0]), int(rgb_center[0,0,1]), int(rgb_center[0,0,2])
        rgb_centers.append((r, g, b))

    # debug print
    print(f"[COLORS] extracted (ordered) centers: {rgb_centers}, counts: {label_counts.tolist()}")
    return rgb_centers

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

# ---------- end new color functions ----------

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

            # Extract top 3 colors (HSV + k-means)
            try:
                top_colors = extract_top_k_colors(pil_img, k=3, attempts=5)
                top_colors_rgb = [f"rgb({c[0]},{c[1]},{c[2]})" for c in top_colors]
                top_colors_hex = [rgb_to_hex(c) for c in top_colors]
            except Exception as ce:
                print("Color extraction failed:", ce)
                top_colors_rgb = [f"rgb({main_rgb[0]},{main_rgb[1]},{main_rgb[2]})"]
                top_colors_hex = [rgb_to_hex(main_rgb)]

            results.append({
                "filename": f.filename,
                "resized_path": save_path,
                "resized_size": f"{pil_img.size[0]}x{pil_img.size[1]}",
                "mode": pil_img.mode,
                "label": label,
                "confidence": round(confidence, 4),
                "main_color": f"rgb({main_rgb[0]},{main_rgb[1]},{main_rgb[2]})",
                "top_colors_rgb": top_colors_rgb,
                "top_colors_hex": top_colors_hex
            })
        except Exception as e:
            results.append({
                "filename": getattr(f, "filename", "unknown"),
                "error": str(e)
            })
    return {"predictions": results}

@app.post("/classify")
async def classify(files: List[UploadFile] = File(...)):
    # Modified: Handle missing model gracefully
    response = []
    for f in files:
        content = await f.read()
        try:
            inp, pil_img, save_path = preprocess_image(content, f.filename, size=INPUT_SIZE)
            
            if interpreter is not None:
                # Model is loaded - use it for classification
                interpreter.set_tensor(input_details[0]['index'], inp)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index']).squeeze()
                probs = softmax(out) if out.ndim == 1 else out
                top_idx = int(np.argmax(probs))
                label = LABELS[top_idx] if top_idx < len(LABELS) else str(top_idx)
            else:
                # Model not loaded - return unknown category
                label = "Unknown"
            
            main_rgb = dominant_color(pil_img)

            # Extract top 3 colors for frontend to use (works without model)
            try:
                top_colors = extract_top_k_colors(pil_img, k=3, attempts=5)
                top_colors_rgb = [f"rgb({c[0]},{c[1]},{c[2]})" for c in top_colors]
            except Exception as ce:
                print("Color extraction failed:", ce)
                top_colors_rgb = [f"rgb({main_rgb[0]},{main_rgb[1]},{main_rgb[2]})"]

            print(f"[CLASSIFY] {f.filename} -> {label}, colors={top_colors_rgb}")
            response.append({
                "filename": f.filename,
                "resized_path": save_path,
                "resized_size": f"{pil_img.size[0]}x{pil_img.size[1]}",
                "mode": pil_img.mode,
                "category": label,
                "dominant_color": f"rgb({main_rgb[0]},{main_rgb[1]},{main_rgb[2]})",
                "top_colors_rgb": top_colors_rgb
            })
        except Exception as e:
            response.append({
                "filename": getattr(f, "filename", "unknown"),
                "error": str(e),
                "category": "error",
                "dominant_color": "rgb(0,0,0)"
            })
    return response

# ---------- Modified: Color matching endpoint (works without model) ----------
@app.post("/color_match")
async def color_match(request: ColorMatchRequest):
    """
    Find clothing items with similar colors to the selected item.
    Returns items of opposite type (if top selected, return bottoms) with color similarity scores.
    This endpoint works independently of the classification model.
    """
    try:
        selected_item = request.selected_item
        all_items = request.all_items
        
        print(f"[COLOR_MATCH] Selected: {selected_item.filename} ({selected_item.type}) - {selected_item.dominant_color}")
        
        # Get opposite type for matching (tops match with bottoms)
        target_type = get_opposite_type(selected_item.type)
        if target_type is None:
            raise HTTPException(status_code=400, detail="Invalid clothing type for matching")
        
        # Filter items by opposite type
        candidate_items = [item for item in all_items if item.type == target_type]
        
        if not candidate_items:
            return {"matches": [], "message": f"No {target_type} items found for matching"}
        
        # Calculate color similarity for each candidate
        matches = []
        for candidate in candidate_items:
            similarity = calculate_color_similarity(
                selected_item.dominant_color, 
                candidate.dominant_color
            )
            
            print(f"[COLOR_MATCH] {candidate.filename} similarity: {similarity:.3f}")
            
            matches.append({
                "filename": candidate.filename,
                "category": candidate.category,
                "type": candidate.type,
                "dominant_color": candidate.dominant_color,
                "similarity": round(similarity, 3)
            })
        
        # Sort by similarity (highest first) and filter good matches (>= 0.6)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        good_matches = [match for match in matches if match["similarity"] >= 0.6]
        
        # If no good matches, return top 3 matches anyway
        if not good_matches:
            good_matches = matches[:3]
        
        print(f"[COLOR_MATCH] Returning {len(good_matches)} matches")
        return {
            "matches": good_matches,
            "selected_item": {
                "filename": selected_item.filename,
                "category": selected_item.category,
                "type": selected_item.type
            },
            "total_candidates": len(candidate_items)
        }
        
    except Exception as e:
        print(f"[COLOR_MATCH] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Color matching failed: {str(e)}")

