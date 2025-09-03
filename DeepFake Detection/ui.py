import os
import shutil
import tempfile

import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- FaceExtractor for MTCNN two-pass detection ---
class FaceExtractor:
    def __init__(self, conf_threshold=0.9, verif_threshold=0.9, min_face_size=40):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            thresholds=[0.7, 0.8, 0.9],
            post_process=False,
            device=self.device
        )
        self.conf_threshold = conf_threshold
        self.verif_threshold = verif_threshold
        self.min_face_size = min_face_size

    def extract_faces(self, video_path, output_dir, frames_per_video=32):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // frames_per_video)
        saved = 0

        with torch.no_grad():
            for fn in range(0, total_frames, frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = self.detector.detect(rgb)
                if boxes is None:
                    continue

                valid = [
                    i for i, (b, p) in enumerate(zip(boxes, probs))
                    if (b[2]-b[0] >= self.min_face_size and
                        b[3]-b[1] >= self.min_face_size and
                        p >= self.conf_threshold)
                ]
                if not valid:
                    continue

                best = valid[np.argmax(probs[valid])]
                x1, y1, x2, y2 = boxes[best].astype(int)
                w, h = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(0.2 * w))
                y1 = max(0, y1 - int(0.2 * h))
                x2 = min(frame.shape[1], x2 + int(0.2 * w))
                y2 = min(frame.shape[0], y2 + int(0.2 * h))
                crop = frame[y1:y2, x1:x2]

                # second-pass verification
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                v_boxes, v_probs = self.detector.detect(crop_rgb)
                if v_boxes is None or np.max(v_probs) < self.verif_threshold:
                    continue

                out_path = os.path.join(output_dir, f"face_{saved:04d}.jpg")
                cv2.imwrite(out_path, crop)
                saved += 1
                if saved >= frames_per_video:
                    break

        cap.release()
        return saved > 0, saved

# --- Load your two models ---
model1 = tf.keras.models.load_model(
    'Final Model/deepfake_detection_model(desnseNet121 Final).h5'
)
model2 = tf.keras.models.load_model(
    'Final Model/deepfake_detection_model(XceptionNet+Bi-LSTM).h5'
)

# --- Ensemble image prediction ---
def predict_deepfake_image(input_image):
    img = Image.fromarray(input_image.astype("uint8")).resize((256, 256))
    arr = image.img_to_array(img)[None, ...] / 255.0

    p1 = float(model1.predict(arr)[0][0])
    p2 = float(model2.predict(arr)[0][0])

    vote1 = 1 if p1 >= 0.5 else 0
    vote2 = 1 if p2 >= 0.5 else 0
    real_votes = vote1 + vote2
    fake_votes = 2 - real_votes

    avg_pred = (p1 + p2) / 2
    final_label = "Real" if avg_pred >= 0.5 else "Fake"

    return (
        f"<div style='font-size:24px;'>"
        f"Final Prediction: <strong>{final_label}</strong><br>"
        "</div>"
    )

# --- Ensemble video prediction ---
extractor = FaceExtractor()
def predict_deepfake_video(video_path):
    tmpdir = tempfile.mkdtemp()
    has_faces, count = extractor.extract_faces(
        video_path, tmpdir, frames_per_video=32
    )
    if not has_faces:
        shutil.rmtree(tmpdir)
        return "<div style='font-size:24px;'> No faces detected in the video.</div>"

    preds1, preds2 = [], []
    for fn in sorted(os.listdir(tmpdir)):
        if not fn.lower().endswith('.jpg'):
            continue
        img = Image.open(os.path.join(tmpdir, fn)).resize((256, 256))
        arr = image.img_to_array(img)[None, ...] / 255.0
        preds1.append(float(model1.predict(arr)[0][0]))
        preds2.append(float(model2.predict(arr)[0][0]))

    shutil.rmtree(tmpdir)

    avg1 = np.mean(preds1)
    avg2 = np.mean(preds2)

    vote1 = 1 if avg1 >= 0.5 else 0
    vote2 = 1 if avg2 >= 0.5 else 0
    real_votes = vote1 + vote2
    fake_votes = 2 - real_votes

    avg_pred = (avg1 + avg2) / 2
    final_label = "Real" if avg_pred >= 0.5 else "Fake"

    return (
        f"<div style='font-size:24px;'>"
        f"Final Prediction: <strong>{final_label}</strong><br>"
        "</div>"
    )

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Deepfake Detection Ensemble")
    gr.Markdown("Upload an **image** or **video** for two‑model majority‑vote prediction.")

    with gr.Tab("Image"):
        img_in = gr.Image(type="numpy", label="Upload Image")
        img_out = gr.HTML(label="Result")
        gr.Button("Detect Image").click(
            predict_deepfake_image, inputs=img_in, outputs=img_out
        )

    with gr.Tab("Video"):
        vid_in = gr.File(
            label="Upload Video",
            file_count="single",
            type="filepath",
            file_types=[".mp4", ".mov", ".avi"]
        )
        vid_out = gr.HTML(label="Result")
        gr.Button("Detect Video").click(
            predict_deepfake_video, inputs=vid_in, outputs=vid_out
        )

demo.launch(share=True)
