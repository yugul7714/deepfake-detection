import cv2
import base64
import requests
from PIL import Image
from io import BytesIO

# Get a frame (from OpenCV or wherever)
frame = cv2.imread("frame.jpg")  # or a video frame
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Convert to base64
buffered = BytesIO()
image.save(buffered, format="JPEG")
encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

# Send to API
response = requests.post("http://127.0.0.1:8000/predict/", json={"frame": encoded})
print(response.json())