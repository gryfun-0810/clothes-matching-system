import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print("Input details:", input_details)

