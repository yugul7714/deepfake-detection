from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model
import uvicorn
import base64

# Load your model
model = load_model("deepfake_detection_model(desnseNet121 Final).h5")
print("✅ Model loaded")

# Create app
app = FastAPI()

# Preprocessing for the model
def preprocess(image: Image.Image):
    image = image.resize((256, 256))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.post("/predict/")
async def predict(request: Request):
    try:
        # Expecting JSON: {"frame": "<base64 string>"}
        body = await request.json()
        base64_data = body["frame"]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        input_arr = preprocess(image)
        prediction = model.predict(input_arr)[0][0]
        label = "Deepfake" if prediction > 0.5 else "Real"

        return JSONResponse(content={
            "label": label,
            "confidence": float(prediction)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
if __name__ == "__main__":
    uvicorn.run("deepfake_api:app", host="0.0.0.0", port=8000, reload=True)