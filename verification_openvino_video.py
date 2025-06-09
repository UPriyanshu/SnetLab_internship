import os
import cv2
import numpy as np
from openvino.runtime import Core

# --------- Model and Video Path ---------
MODEL_PATH = "/home/priyanshuuniyal/yolov7/yolov7_openvino/yolov7.xml"
BIN_PATH = "/home/priyanshuuniyal/yolov7/yolov7_openvino/yolov7.bin"
VIDEO_PATH = "/home/priyanshuuniyal/yolov7/inference/video/vide.mp4"
OUTPUT_PATH = "openvino_video_output.mp4"

# --------- Check If Model Files Exist ---------
if not os.path.isfile(MODEL_PATH) or not os.path.isfile(BIN_PATH):
    raise FileNotFoundError(f"Model files not found. Make sure {MODEL_PATH} and {BIN_PATH} exist.")

# --------- Load Model ---------
core = Core()
model = core.read_model(MODEL_PATH)
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --------- Class Names ---------
CLASSES = [...]  # (Use the same CLASSES list from your script for brevity)

# --------- Detection Parameters ---------
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# --------- Load Video ---------
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --------- Output Writer ---------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --------- Process Each Frame ---------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    img = resized.transpose(2, 0, 1) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Inference
    outputs = compiled_model([img])[output_layer]

    boxes, confidences, class_ids = [], [], []
    rows = outputs.shape[1]

    for i in range(rows):
        row = outputs[0][i]
        obj_conf = row[4]
        if obj_conf < CONF_THRESHOLD:
            continue

        class_probs = row[5:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id] * obj_conf
        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, w, h = row[0:4]
        x1 = int((cx - w / 2) * orig_w / 640)
        y1 = int((cy - h / 2) * orig_h / 640)
        x2 = int((cx + w / 2) * orig_w / 640)
        y2 = int((cy + h / 2) * orig_h / 640)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # NMS and Draw
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        class_id = class_ids[i]
        if class_id < len(CLASSES):
            label = f"{CLASSES[class_id]} {confidences[i]:.2f}"
        else:
            label = f"class {class_id} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("OpenVINO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------- Release Resources ---------
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Video inference completed and saved to", OUTPUT_PATH)

