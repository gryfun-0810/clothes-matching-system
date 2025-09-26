import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

MODEL_PATH = "model/model.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_inference(img_path):
    img = Image.open(img_path).convert("RGB").resize((224,224))
    arr = np.array(img).astype(np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index']).squeeze()
    print(img_path, out)

run_inference("temp_images/img1_resized.png")
run_inference("temp_images/img2_resized.png")
run_inference("temp_images/img3_resized.png")


