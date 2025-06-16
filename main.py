import cv2
import torch
import time
from playsound import playsound
import threading
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Initialize DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# Configuration
TARGET_CLASS = 'bottle'
MISSING_THRESHOLD = 3  # seconds
ALERT_SOUND = 'alert.mp3' 
CONFIDENCE_THRESHOLD = 0.3

# COCO class names
CLASS_NAMES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# State variables
last_detection_time = None
alert_playing = False
alert_thread = None
alert_triggered = False
stop_alert_flag = False  # New flag to signal alert to stop

def play_alert():
    global alert_playing, alert_triggered, stop_alert_flag
    try:
        # Play alert in a loop until stopped
        while not stop_alert_flag:
            playsound(ALERT_SOUND)
            time.sleep(0.1)  # Small delay to allow for interruption
    except:
        pass
    alert_playing = False
    alert_triggered = True
    stop_alert_flag = False  # Reset for next time

def stop_alert():
    global stop_alert_flag, alert_playing, alert_triggered
    stop_alert_flag = True
    alert_playing = False
    alert_triggered = False

def detect_objects(image):
    img = Image.fromarray(image)
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > CONFIDENCE_THRESHOLD
    
    boxes = rescale_bboxes(outputs['pred_boxes'][0, keep], img.size)
    labels = probas[keep].argmax(dim=1)
    scores = probas[keep].max(dim=1).values
    
    return boxes, labels, scores

def draw_detections(image, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        box = box.cpu().numpy()
        label = label.item()
        score = score.item()
        
        if CLASS_NAMES[label] != TARGET_CLASS:
            continue
            
        color = (0, 255, 0)
        cv2.rectangle(image, 
                     (int(box[0]), int(box[1])),
                     (int(box[2]), int(box[3])),
                     color, 2)
        cv2.putText(image, f'{CLASS_NAMES[label]}: {score:.2f}',
                   (int(box[0]), int(box[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Main loop
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, scores = detect_objects(rgb_frame)
        
        target_detected = False
        for label in labels:
            if CLASS_NAMES[label.item()] == TARGET_CLASS:
                target_detected = True
                break
        
        if target_detected:
            last_detection_time = time.time()
            if alert_playing or alert_triggered:
                stop_alert()  # Immediately stop alert when bottle is detected
        else:
            if last_detection_time is None:
                last_detection_time = time.time()
            else:
                time_missing = time.time() - last_detection_time
                if time_missing > MISSING_THRESHOLD and not alert_playing and not alert_triggered:
                    alert_thread = threading.Thread(target=play_alert)
                    alert_thread.start()
                    alert_playing = True
        
        frame = draw_detections(frame, boxes, labels, scores)
        
        if target_detected:
            status = "Present"
            color = (0, 255, 0)
        elif last_detection_time is None:
            status = "Initializing..."
            color = (255, 255, 0)
        else:
            time_missing = time.time() - last_detection_time
            if time_missing > MISSING_THRESHOLD:
                status = "ALERT: Bottle missing!"
                color = (0, 0, 255)
            else:
                status = f"Missing in: {MISSING_THRESHOLD - time_missing:.1f}s"
                color = (0, 165, 255)
        
        cv2.putText(frame, f'Bottle: {status}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Bottle Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_alert()
    cap.release()
    cv2.destroyAllWindows()
    if alert_thread is not None:
        alert_thread.join()