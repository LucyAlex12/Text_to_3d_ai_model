from diffusers import StableDiffusionPipeline
from rembg import remove
from PIL import Image
from io import BytesIO

import torch
import subprocess
import os
import time
import sys
import json
import threading

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD MODEL ONCE
# =========================================================

print("Loading Stable Diffusion...")

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

pipe.safety_checker = None

pipe.enable_attention_slicing()

print("Stable Diffusion loaded")

# =========================================================
# PROMPT ENGINEERING
# =========================================================

def build_prompt(user_prompt):

    return f"""
    {user_prompt},

    isolated object,
    single object only,
    centered object,
    full object visible,
    orthographic view,
    front view,
    white background,
    no environment,
    no room,
    no walls,
    no floor,
    no shadows,
    no reflections,
    studio render,
    product render,
    clean silhouette,
    low poly,
    game asset,
    3d asset
    """

NEGATIVE_PROMPT = """
background,
room,
floor,
wall,
environment,
scene,
multiple objects,
cropped,
cut off,
shadow,
reflection,
realistic room,
complex background,
table,
bad anatomy,
deformed,
blurry
"""

# =========================================================
# PROGRESS SYSTEM
# =========================================================

current_progress = 1

def update_progress(step, percent):

    with open(
        os.path.join(OUTPUT_DIR, "progress.json"),
        "w"
    ) as f:

        json.dump({
            "step": step,
            "progress": int(percent)
        }, f)

def smooth_progress():

    global current_progress

    while current_progress < 95:

        time.sleep(2)

        current_progress += 2

        update_progress(
            "Generating 3D model...",
            current_progress
        )

# =========================================================
# MAIN
# =========================================================

def main():

    global current_progress

    user_prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "wooden chair"
    )

    prompt = build_prompt(user_prompt)

    # =====================================================
    # JOB FOLDER
    # =====================================================

    job_id = str(int(time.time()))

    job_dir = os.path.join(
        OUTPUT_DIR,
        job_id
    )

    os.makedirs(job_dir, exist_ok=True)

    image_path = os.path.join(
        job_dir,
        "input.png"
    )

    # =====================================================
    # START PROGRESS
    # =====================================================

    update_progress(
        "Generating 3D model...",
        1
    )

    thread = threading.Thread(
        target=smooth_progress
    )

    thread.start()

    # =====================================================
    # GENERATE IMAGE
    # =====================================================

    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=9,
        width=512,
        height=512
    )

    image = result.images[0]

    print("Image generated")

# =====================================================
# REMOVE BACKGROUND PROPERLY
# =====================================================

image = image.convert("RGBA")

buffer = BytesIO()
image.save(buffer, format="PNG")

input_bytes = buffer.getvalue()

# rembg remove
output_bytes = remove(input_bytes)

# reopen processed image
processed = Image.open(BytesIO(output_bytes)).convert("RGBA")

# CREATE PURE TRANSPARENT BACKGROUND
new_data = []

for item in processed.getdata():

    r, g, b, a = item

    # remove near-white background completely
    if r > 240 and g > 240 and b > 240:
        new_data.append((255, 255, 255, 0))

    else:
        new_data.append((r, g, b, 255))

processed.putdata(new_data)

# CROP OBJECT TIGHTLY
bbox = processed.getbbox()

processed = processed.crop(bbox)

# RESIZE TO SQUARE
processed.thumbnail((512, 512))

final_img = Image.new(
    "RGBA",
    (512, 512),
    (255, 255, 255, 0)
)

offset_x = (512 - processed.width) // 2
offset_y = (512 - processed.height) // 2

final_img.paste(
    processed,
    (offset_x, offset_y),
    processed
)

# SAVE FINAL
final_img.save(image_path)

print("Background fully removed")

    # =====================================================
    # RUN TRIPOSR
    # =====================================================

    tripo_result = subprocess.run([
        sys.executable,
        os.path.join(
            BASE_DIR,
            "TripoSR",
            "run.py"
        ),
        image_path,
        "--output-dir",
        job_dir,
        "--model-save-format",
        "glb",
        "--bake-texture"
    ])

    current_progress = 95

    # =====================================================
    # FAILED
    # =====================================================

    if tripo_result.returncode != 0:

        update_progress(
            "Generation failed",
            0
        )

        print("Generation failed")

        sys.exit(1)

    # =====================================================
    # FIND GLB
    # =====================================================

    glb_path = None

    for root, dirs, files in os.walk(job_dir):

        for file in files:

            if file.endswith(".glb"):

                glb_path = os.path.join(root, file)

    if glb_path is None:

        update_progress(
            "No GLB generated",
            0
        )

        print("No GLB file found")

        sys.exit(1)

    print("Generated GLB:", glb_path)

    # =====================================================
    # FINALIZING
    # =====================================================

    update_progress(
        "Finalizing...",
        98
    )

    time.sleep(1)

    update_progress(
        "Done!",
        100
    )

    print("DONE!")

# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()