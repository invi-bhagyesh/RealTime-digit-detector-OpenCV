import cv2
import numpy as np
import torch
from model import MNISTCNN  # Import your CNN model from model.py
import torch.nn.functional as F  # Import softmax function

# Load trained model
model = MNISTCNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device("cpu")), strict=False)
model.eval()

hsv_value = np.load('hsv_value.npy')


# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

kernel = np.ones((5, 5), np.uint8)  # Kernel for morphological operations
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Drawing canvas

x1, y1 = 0, 0
noise_thresh = 800  # Lower threshold to capture smaller digits



# Function to preprocess the drawn digit for model input
def preprocess_digit(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)  # Ensure black on white
    """
    Change from threshold from 50 to 30
    """

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)  # Reduce holes
    thresh = cv2.erode(thresh, kernel, iterations=1)

    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)  # Better resize
    normalized = resized.astype(np.float32) / 255.0  # Normalize

    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to tensor
    return tensor



# Main OpenCV loop
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip for mirror effect

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array(hsv_value[0], dtype=np.uint8)  
    upper_range = np.array(hsv_value[1], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Remove noise
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Reduce flickering by blurring

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small noise by setting a minimum contour area
    valid_contours = [c for c in contours if cv2.contourArea(c) > noise_thresh]
    if valid_contours:
        c = max(valid_contours, key=cv2.contourArea)  # Select the largest valid contour
        x2, y2, w, h = cv2.boundingRect(c)

    # If a valid contour is detected, draw on the canvas
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        if x1 == 0 and y1 == 0:
            x1, y1 = x2, y2
        else:
            # Prevent sudden jumps (set max distance threshold)
            if abs(x2 - x1) < 50 and abs(y2 - y1) < 50:  # Adjusted thresholds for smoother drawing
                canvas = cv2.line(canvas, (x1, y1), (x2, y2), [255, 255, 255], 20)  # Increased thickness to 10

            x1, y1 = x2, y2  # Update pen position only if movement is valid
    else:
        x1, y1 = 0, 0

    # Apply Gaussian blur to canvas for additional smoothness
    canvas = cv2.GaussianBlur(canvas, (5, 5), 0)

    frame = cv2.add(canvas, frame)

    key = cv2.waitKey(1) & 0xFF  # First read the key

    if key == ord('p'):  
        digit_tensor = preprocess_digit(canvas)  
        with torch.no_grad():
            prediction = model(digit_tensor)  
            probabilities = F.softmax(prediction, dim=1)  # Convert logits to probabilities
            predicted_digit = torch.argmax(probabilities).item()
            confidence = probabilities[0, predicted_digit].item()  # Get confidence score

        print(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2%})")

        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Clear canvas after prediction

    frame = cv2.resize(frame, (canvas.shape[1], canvas.shape[0]))  # Match canvas size
    stacked = np.hstack((canvas, frame))
    cv2.imshow('Screen_Pen', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    if key == ord('q'):  
        break
    elif key == ord('c'):  
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)  # Clear canvas manually

cap.release()  # Release before destroying windows
cv2.destroyAllWindows()