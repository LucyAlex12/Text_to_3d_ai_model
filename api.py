import os
import uuid
import torch
import uvicorn

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from diffusers import StableDiffusionPipeline
from PIL import Image

# =========================
# TRIPOSR IMPORT
# =========================

import sys
sys.path.append("./TripoSR")

from tsr.system import TSR

# =========================
# APP
# =========================

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD SD PIPELINE
# =========================

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to(device)

# =========================
# LOAD TRIPOSR
# =========================

model = TSR.from_pretrained(
    "./TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

model.renderer.set_chunk_size(8192)
model.to(device)

print("TripoSR loaded successfully")

# =========================
# GLOBAL PROGRESS
# =========================

progress_data = {
    "progress": 0,
    "status": "idle"
}

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/progress")
def progress():
    return JSONResponse(progress_data)

@app.get("/generate3d")
def generate3d(prompt: str):

    try:

        progress_data["progress"] = 5
        progress_data["status"] = "Generating 3d model"

        # =========================
        # UNIQUE JOB ID
        # =========================

        uid = str(uuid.uuid4())[:8]

        out_dir = os.path.join(OUTPUT_DIR, uid)
        os.makedirs(out_dir, exist_ok=True)

        image_path = os.path.join(out_dir, "input.png")
        glb_path = os.path.join(out_dir, "mesh.glb")

        # =========================
        # GENERATE IMAGE
        # =========================

        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        image.save(image_path)

        print("Generated image:", image_path)

        progress_data["progress"] = 40
        progress_data["status"] = "Generating 3D model"

        # =========================
        # LOAD IMAGE
        # =========================

        image = Image.open(image_path).convert("RGB")

        # =========================
        # RUN TRIPOSR
        # =========================

        scene_codes = model([image], device=device)

        meshes = model.extract_mesh(
            scene_codes,
            resolution=256,
            has_vertex_color=True
        )

        mesh = meshes[0]

        # =========================
        # EXPORT TEXTURED GLB
        # =========================

        mesh.export(glb_path)

        print("Saved GLB:", glb_path)

        progress_data["progress"] = 100
        progress_data["status"] = "Done"

        return {
            "model_url": f"/outputs/{uid}/mesh.glb",
            "image_url": f"/outputs/{uid}/input.png"
        }

    except Exception as e:

        progress_data["progress"] = 0
        progress_data["status"] = f"Error: {str(e)}"

        print(e)

        return {
            "error": str(e)
        }

# =========================
# RUN
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )