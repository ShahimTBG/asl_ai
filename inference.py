import os
import random
import numpy as np
from PIL import Image
import platform

# Use Edge TPU if available
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    print(" Using Edge TPU delegate.")
    interpreter = Interpreter(
        model_path="asl_model.tflite",
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    using_tpu = True
except Exception as e:
    import tensorflow as tf
    print(" Edge TPU not available, falling back to CPU.")
    interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
    using_tpu = False

# Label list must match training order
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Load and allocate model
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Path to test images
test_path = "asl_alphabet_test/asl_alphabet_test"
image_files = [f for f in os.listdir(test_path) if f.endswith(".jpg") or f.endswith(".png")]

# Track accuracy
correct = 0
total = 0

# Run prediction on random N images
N = 10  # change to 100+ for deeper accuracy test
samples = random.sample(image_files, min(N, len(image_files)))

for img_file in samples:
    label_from_filename = img_file.split("_")[0].upper()
    image_path = os.path.join(test_path, img_file)

    # Preprocess
    image = Image.open(image_path).convert("RGB")
    image = image.resize((64, 64))
    image_array = np.array(image).astype(np.float32) / 255.0
    input_data = np.expand_dims(image_array, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction_index = int(np.argmax(output_data))
    prediction_label = class_labels[prediction_index]
    confidence = float(np.max(output_data))

    # Check correctness
    is_correct = (prediction_label == label_from_filename)
    if is_correct:
        correct += 1
    total += 1

    print(f"\n Image: {img_file}")
    print(f" Actual: {label_from_filename}")
    print(f" Predicted: {prediction_label} (Index {prediction_index})")
    print(f" Confidence: {confidence:.4f}")
    print(f"{' Correct' if is_correct else ' Wrong'}")

# Final accuracy
accuracy = correct / total if total else 0
print(f"\n Inference Device: {'Edge TPU' if using_tpu else 'CPU'}")
print(f" Accuracy on {total} samples: {accuracy*100:.2f}%")