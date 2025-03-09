import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import torchvision.transforms as T

# Load YOLO model
model = YOLO('best.pt')  # Replace with your trained model

# Load depth estimation model (MiDaS)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # Lightweight model
midas.eval()

# Depth estimation transformations
transform = Compose([
    T.Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize camera
cap = cv2.VideoCapture(0)  # Change to video file if needed

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to PIL Image for MiDaS
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Depth estimation (Monocular)
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        depth_map = midas(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Run YOLO for object detection
    results = model(frame)
    detections = results[0].boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
        depth = np.mean(depth_map[y1:y2, x1:x2])  # Get depth estimate

        # Convert pixel height to real-world height (assuming known focal length)
        pixel_height = y2 - y1
        focal_length = 500  # Adjust based on calibration
        real_height_cm = (pixel_height * depth) / focal_length  # Approximate conversion

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Height: {real_height_cm:.2f} cm", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
