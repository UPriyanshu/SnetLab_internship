import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import sys

# === CONFIG ===
engine_path = "/home/priyanshuuniyal/yolov7/yolov7.trt"
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
CONF_THRESHOLD = 0.3
class_file = "/home/priyanshuuniyal/yolov7/classes.txt"

# Load class names
with open(class_file, 'r') as f:
    class_names = [c.strip() for c in f.readlines()]

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        print(f"[INFO] Loading TensorRT engine from {engine_path}...")
        return runtime.deserialize_cuda_engine(f.read())

def preprocess(image):
    image_resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_transposed = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    image_normalized = image_transposed / 255.0
    return np.expand_dims(image_normalized, axis=0)

def postprocess(outputs, img_shape, orig_img):
    output = outputs[0]
    h_ratio, w_ratio = orig_img.shape[0] / INPUT_HEIGHT, orig_img.shape[1] / INPUT_WIDTH

    boxes = []
    confidences = []
    class_ids = []

    for det in output:
        conf = det[4]
        if conf < CONF_THRESHOLD:
            continue

        scores = det[5:]
        class_id = np.argmax(scores)
        if scores[class_id] < CONF_THRESHOLD:
            continue

        cx, cy, w, h = det[0:4]
        x = int((cx - w / 2) * w_ratio)
        y = int((cy - h / 2) * h_ratio)
        width = int(w * w_ratio)
        height = int(h * h_ratio)

        boxes.append([x, y, width, height])
        confidences.append(float(conf))
        class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.5)
    results = []
    for i in indices:
        i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
        box = boxes[i]
        x, y, w, h = box
        results.append((class_ids[i], confidences[i], (x, y, x + w, y + h)))
    return results


def draw_boxes(img, detections):
    for class_id, conf, box in detections:
        x1, y1, x2, y2 = box
        label = f"{class_names[class_id]}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def infer(engine, input_image):
    context = engine.create_execution_context()

    input_shape = (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    output_shape = (1, 25200, 85)  # YOLOv7 output for 640x640 with COCO

    input_size = int(np.prod(input_shape)) * np.float32().nbytes
    output_size = int(np.prod(output_shape)) * np.float32().nbytes

    # Allocate device memory
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    bindings = [int(d_input), int(d_output)]

    h_input = np.ascontiguousarray(preprocess(input_image), dtype=np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    # Copy input to device
    cuda.memcpy_htod(d_input, h_input)

    # Run inference
    context.execute_v2(bindings)

    # Copy output back to host
    cuda.memcpy_dtoh(h_output, d_output)

    return postprocess(h_output, input_image.shape, input_image)

def run(source):
    engine = load_engine(engine_path)

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source.endswith(".jpg") or source.endswith(".png"):
        img = cv2.imread(source)
        detections = infer(engine, img)
        img = draw_boxes(img, detections)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    else:
        cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        detections = infer(engine, frame)
        end = time.time()

        frame = draw_boxes(frame, detections)
        fps = 1.0 / (end - start)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # cv2.imshow("TensorRT YOLOv7", frame)
        resized_frame = cv2.resize(frame, (1536, 864))  # width x height
        cv2.imshow("TensorRT YOLOv7", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_trt.py [image.jpg | video.mp4 | webcam]")
        sys.exit(1)
    run(sys.argv[1])
