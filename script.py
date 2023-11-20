import cv2
import numpy as np
import sys

# Load YOLOv3 model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg.1', 'yolov3.weights')

# Load COCO class names
classes = []
with open('coco.names.1', 'r') as f:
    classes = f.read().strip().split('\n')

# Set the video capture source (0 for default camera, or provide the path to a video file)
cap = cv2.VideoCapture(sys.argv[1])

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Get the video frame dimensions
    height, width, _ = frame.shape

    # Convert the frame to a blob and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label on the frame
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('img',frame)
   
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

