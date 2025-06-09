import os
import cv2
import numpy as np
from openvino.runtime import Core

# --------- Model Path ---------
MODEL_PATH = "/home/priyanshuuniyal/yolov7/yolov7_openvino/yolov7.xml"
BIN_PATH = "/home/priyanshuuniyal/yolov7/yolov7_openvino/yolov7.bin"

# --------- Check Model Files ---------
if not os.path.isfile(MODEL_PATH) or not os.path.isfile(BIN_PATH):
    raise FileNotFoundError("Model files not found!")

# --------- Load OpenVINO Model ---------
core = Core()
model = core.read_model(MODEL_PATH)
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# --------- Class Labels ---------
CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# --------- Start Webcam ---------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    img = resized.transpose(2, 0, 1) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # --------- Inference ---------
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{CLASSES[class_ids[i]]} {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # --------- Display Result ---------
    cv2.imshow("YOLOv7 OpenVINO - Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

