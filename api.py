import os
import re
import shutil
import sys
import uuid
from typing import Optional

import numpy as np
import rembg
import torch
import trimesh
import uvicorn

from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from diffusers import DPMSolverMultistepScheduler
    from diffusers import StableDiffusionPipeline
    from diffusers import StableDiffusionXLPipeline
except ImportError:
    DPMSolverMultistepScheduler = None
    StableDiffusionPipeline = None
    StableDiffusionXLPipeline = None

# =====================================================
# TRIPOSR
# =====================================================

sys.path.append("./TripoSR")

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

# =====================================================
# APP
# =====================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# SETTINGS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

SDXL_MODEL_ID = os.getenv(
    "SDXL_MODEL_ID",
    "stabilityai/stable-diffusion-xl-base-1.0"
)

SD15_MODEL_ID = os.getenv(
    "SD15_MODEL_ID",
    "runwayml/stable-diffusion-v1-5"
)

IMAGE_MODEL_KIND = os.getenv(
    "IMAGE_MODEL_KIND",
    "sdxl" if torch.cuda.is_available() else "sd15"
)

FOREGROUND_RATIO = float(
    os.getenv("TRIPOSR_FOREGROUND_RATIO", "0.85")
)

DEFAULT_MC_RESOLUTION = int(
    os.getenv("TRIPOSR_MC_RESOLUTION", "256")
)

DEFAULT_MESH_THRESHOLD = float(
    os.getenv("TRIPOSR_ISOSURFACE_THRESHOLD", "28.0")
)

MAX_MC_RESOLUTION = int(
    os.getenv("TRIPOSR_MAX_MC_RESOLUTION", "320")
)

DEFAULT_MESH_QUALITY = os.getenv(
    "TRIPOSR_MESH_QUALITY",
    "balanced"
).lower()

MESH_QUALITY_PRESETS = {
    "fast": {
        "resolution": 192,
        "smooth_iterations": 4,
        "threshold": DEFAULT_MESH_THRESHOLD,
    },
    "balanced": {
        "resolution": DEFAULT_MC_RESOLUTION,
        "smooth_iterations": 8,
        "threshold": DEFAULT_MESH_THRESHOLD,
    },
    "high": {
        "resolution": 320,
        "smooth_iterations": 10,
        "threshold": DEFAULT_MESH_THRESHOLD,
    },
}

MESH_BACK_DEPTH_CONSTRAINT_ENABLED = os.getenv(
    "TRIPOSR_BACK_DEPTH_CONSTRAINT",
    "1"
) != "0"

MESH_BACK_DEPTH_STRENGTH = float(
    os.getenv("TRIPOSR_BACK_DEPTH_STRENGTH", "0.45")
)

SDXL_WIDTH = int(
    os.getenv("SDXL_WIDTH", "512" if not torch.cuda.is_available() else "1024")
)

SDXL_HEIGHT = int(
    os.getenv("SDXL_HEIGHT", "512" if not torch.cuda.is_available() else "1024")
)

SDXL_STEPS = int(
    os.getenv("SDXL_STEPS", "16" if not torch.cuda.is_available() else "35")
)

SDXL_GUIDANCE_SCALE = float(
    os.getenv("SDXL_GUIDANCE_SCALE", "7.0" if not torch.cuda.is_available() else "6.5")
)

KEEP_MAIN_FOREGROUND = os.getenv(
    "TRIPOSR_KEEP_MAIN_FOREGROUND",
    "1"
) != "0"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =====================================================
# STATIC FILES
# =====================================================

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

# =====================================================
# DEVICE
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch_dtype = (
    torch.float16
    if device == "cuda"
    else torch.float32
)

print("Using device:", device)

# =====================================================
# LOAD TRIPOSR
# =====================================================

triposr_model = TSR.from_pretrained(
    "./TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

triposr_model.renderer.set_chunk_size(8192)

triposr_model.to(device)

print("TripoSR loaded")

image_pipe = None

rembg_session = rembg.new_session()

# =====================================================
# PROGRESS
# =====================================================

progress_data = {
    "progress": 0,
    "status": "Idle"
}

# =====================================================
# PROMPTS
# =====================================================

STYLE_PROMPT_RULES = [
    (
        ["low poly", "low-poly"],
        "clean intentional low-poly style, simple faceted forms, stylized polygonal 3D asset, not a rough broken mesh",
    ),
    (
        ["voxel"],
        "clean voxel style, blocky toy-like silhouette, crisp square forms, solid 3D asset",
    ),
    (
        ["pixel art", "pixel-art"],
        "pixel-art inspired 3D toy style, simplified crisp forms, clear object silhouette",
    ),
    (
        ["sketch"],
        "sketch-inspired concept render, clear solid object shape, readable 3D asset silhouette",
    ),
]


IDENTITY_PROMPT_RULES = [
    (
        ["robot", "android", "cyborg", "mech"],
        "clearly robotic subject, visible mechanical joints, robot limbs, synthetic body panels, mechanical details, not just an ordinary human or fashion doll",
    ),
    (
        ["chair", "stool"],
        "recognizable chair structure, clear seat, clear backrest, clear legs, no extra duplicate chair",
    ),
    (
        ["table", "desk"],
        "recognizable table structure, clear tabletop, clear supporting legs, no extra duplicate table",
    ),
]


BASE_NEGATIVE_TERMS = [
    "background scene",
    "environment",
    "room",
    "floor",
    "table",
    "ground",
    "wall",
    "shadow",
    "reflection",
    "multiple objects",
    "extra objects",
    "duplicate object",
    "second object",
    "object pair",
    "two views",
    "reference sheet",
    "inset image",
    "split screen",
    "comparison image",
    "extra limbs",
    "cropped",
    "cut off",
    "close up",
    "blurry",
    "motion blur",
    "dark image",
    "black background",
    "transparent object",
    "holes through the object",
    "rough surface",
    "jagged edges",
    "wireframe",
    "mesh lines",
    "grid pattern",
    "checkerboard pattern",
    "crosshatch",
    "noisy texture",
    "busy texture",
    "scan lines",
    "glitch",
    "low poly",
    "voxel",
    "pixel art",
    "sketch",
    "text",
    "logo",
    "watermark",
    "person holding object",
    "complex scenery",
]


FRONT_BIASED_MESH_TERMS = [
    "person",
    "human",
    "woman",
    "man",
    "female",
    "male",
    "girl",
    "boy",
    "character",
    "figure",
    "figurine",
    "statue",
    "mannequin",
    "doll",
    "barbie",
    "toy",
    "robot",
    "android",
    "cyborg",
    "mech",
]


def prompt_has_term(prompt: str, term: str) -> bool:

    normalized_prompt = prompt.lower().replace("-", " ")
    normalized_term = term.lower().replace("-", " ")

    pattern = r"\b" + re.escape(normalized_term).replace(r"\ ", r"\s+") + r"s?\b"

    return re.search(
        pattern,
        normalized_prompt
    ) is not None


def collect_prompt_rules(user_prompt: str, rules: list[tuple[list[str], str]]) -> list[str]:

    collected = []

    for terms, text in rules:

        if any(prompt_has_term(user_prompt, term) for term in terms):

            collected.append(
                text
            )

    return collected


def build_prompt(user_prompt: str) -> str:

    subject = user_prompt.strip()

    identity_rules = collect_prompt_rules(
        subject,
        IDENTITY_PROMPT_RULES
    )

    style_rules = collect_prompt_rules(
        subject,
        STYLE_PROMPT_RULES
    )

    emphasis = "\n    ".join(
        identity_rules + style_rules
    )

    return f"""
    exact subject: {subject},
    a single {subject},
    must visibly match the exact subject: {subject},
    one complete object only,
    centered in frame,
    full object visible,
    front three-quarter view,
    clean readable silhouette,
    {emphasis}
    compact solid form,
    smooth clean surfaces,
    simple clear geometry,
    crisp object boundaries,
    solid material,
    matte product render,
    sharp focus,
    neutral gray studio background,
    product render,
    orthographic camera,
    evenly lit,
    high quality
    """


def build_negative_prompt(user_prompt: str) -> str:

    negative_terms = [
        term
        for term in BASE_NEGATIVE_TERMS
        if not prompt_has_term(user_prompt, term)
    ]

    return ",\n".join(
        negative_terms
    )

# =====================================================
# MODEL HELPERS
# =====================================================

def get_image_pipe():

    global image_pipe

    if image_pipe is not None:

        return image_pipe

    if IMAGE_MODEL_KIND == "sdxl":

        pipeline_cls = StableDiffusionXLPipeline

        model_id = SDXL_MODEL_ID

    else:

        pipeline_cls = StableDiffusionPipeline

        model_id = SD15_MODEL_ID

    if pipeline_cls is None:

        raise RuntimeError(
            "diffusers is not installed. Install the packages in requirements.txt first."
        )

    print(f"Loading image model: {model_id}")

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    )

    if DPMSolverMultistepScheduler is not None:

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

    pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    try:

        pipe.enable_vae_slicing()

    except Exception:

        pass

    if hasattr(pipe, "safety_checker"):

        pipe.safety_checker = None

    image_pipe = pipe

    print("Image model loaded")

    return image_pipe


def generate_source_image(prompt: str, seed: Optional[int]) -> tuple[Image.Image, int]:

    pipe = get_image_pipe()

    if seed is None:

        seed = int.from_bytes(
            os.urandom(4),
            "big"
        )

    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    result = pipe(
        prompt=build_prompt(prompt),
        negative_prompt=build_negative_prompt(prompt),
        num_inference_steps=SDXL_STEPS,
        guidance_scale=SDXL_GUIDANCE_SCALE,
        width=SDXL_WIDTH,
        height=SDXL_HEIGHT,
        generator=generator
    )

    return result.images[0], seed


def load_example_image(name: str = "robot.png") -> Image.Image:

    image_path = os.path.join(
        BASE_DIR,
        "TripoSR",
        "examples",
        name
    )

    return Image.open(image_path)

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def composite_on_gray(image: Image.Image) -> Image.Image:

    image = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    image = (
        image[:, :, :3]
        * image[:, :, 3:4]
        + (1 - image[:, :, 3:4]) * 0.5
    )

    return Image.fromarray(
        (image * 255.0).astype(np.uint8)
    )


def keep_main_foreground_object(image: Image.Image) -> Image.Image:

    if not KEEP_MAIN_FOREGROUND or cv2 is None:

        return image

    if image.mode != "RGBA":

        return image

    arr = np.array(
        image
    )

    alpha = arr[:, :, 3]

    mask = (
        alpha > 10
    ).astype(np.uint8)

    if mask.max() == 0:

        return image

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        8
    )

    if count <= 2:

        return image

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_label = int(
        np.argmax(areas) + 1
    )

    largest_area = float(
        stats[largest_label, cv2.CC_STAT_AREA]
    )

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    pad = int(
        max(w, h) * 0.18 + 24
    )

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(mask.shape[1], x + w + pad)
    y2 = min(mask.shape[0], y + h + pad)

    keep_mask = labels == largest_label

    for label in range(1, count):

        if label == largest_label:

            continue

        area = float(
            stats[label, cv2.CC_STAT_AREA]
        )

        cx = stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] / 2
        cy = stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] / 2

        is_near_main_object = (
            x1 <= cx <= x2
            and y1 <= cy <= y2
        )

        is_meaningful_detail = area >= largest_area * 0.002

        if is_near_main_object and is_meaningful_detail:

            keep_mask |= labels == label

    arr[:, :, 3] = np.where(
        keep_mask,
        arr[:, :, 3],
        0
    ).astype(np.uint8)

    return Image.fromarray(
        arr
    )


def preprocess_for_triposr(image: Image.Image) -> Image.Image:

    if image.mode != "RGBA":

        image = image.convert("RGB")

    image = remove_background(
        image,
        rembg_session
    )

    image = keep_main_foreground_object(
        image
    )

    image = resize_foreground(
        image,
        FOREGROUND_RATIO
    )

    return composite_on_gray(image)

# =====================================================
# MESH HELPERS
# =====================================================

def resolve_mesh_settings(quality: str, mc_resolution: Optional[int]) -> dict:

    quality_key = (quality or DEFAULT_MESH_QUALITY).lower()

    if quality_key not in MESH_QUALITY_PRESETS:

        quality_key = DEFAULT_MESH_QUALITY

    if quality_key not in MESH_QUALITY_PRESETS:

        quality_key = "balanced"

    settings = dict(
        MESH_QUALITY_PRESETS[quality_key]
    )

    if mc_resolution is not None:

        settings["resolution"] = mc_resolution

    settings["resolution"] = max(
        64,
        min(
            int(settings["resolution"]),
            MAX_MC_RESOLUTION
        )
    )

    settings["quality"] = quality_key

    return settings


def keep_primary_components(mesh, min_face_ratio: float = 0.01):

    try:

        parts = mesh.split(
            only_watertight=False
        )

    except Exception:

        return mesh

    if len(parts) <= 1:

        return mesh

    largest_face_count = max(
        len(part.faces)
        for part in parts
    )

    kept_parts = [
        part
        for part in parts
        if len(part.faces) >= largest_face_count * min_face_ratio
    ]

    if not kept_parts:

        return mesh

    if len(kept_parts) == 1:

        return kept_parts[0]

    return trimesh.util.concatenate(
        kept_parts
    )


def smooth_mesh(mesh, iterations: int):

    if iterations <= 0:

        return mesh

    try:

        trimesh.smoothing.filter_taubin(
            mesh,
            lamb=0.45,
            nu=-0.53,
            iterations=iterations
        )

    except Exception:

        try:

            trimesh.smoothing.filter_laplacian(
                mesh,
                lamb=0.25,
                iterations=max(1, iterations // 2),
                volume_constraint=True
            )

        except Exception:

            pass

    return mesh


def prompt_prefers_front_biased_mesh(prompt: str) -> bool:

    return any(
        prompt_has_term(prompt, term)
        for term in FRONT_BIASED_MESH_TERMS
    )


def constrain_unseen_back_depth(mesh, prompt: str):

    if not MESH_BACK_DEPTH_CONSTRAINT_ENABLED:

        return mesh

    strength = max(
        0.0,
        min(
            float(MESH_BACK_DEPTH_STRENGTH),
            0.85
        )
    )

    if strength <= 0.0:

        return mesh

    try:

        extents = np.asarray(
            mesh.extents,
            dtype=np.float64
        )

    except Exception:

        return mesh

    if extents.shape[0] < 3 or np.any(extents <= 0):

        return mesh

    width, height, depth = extents[:3]

    is_tall_front_view = (
        height >= max(width, depth) * 1.10
        and depth <= max(width * 0.95, height * 0.60)
    )

    if not (
        prompt_prefers_front_biased_mesh(prompt)
        and is_tall_front_view
    ):

        return mesh

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64
    ).copy()

    if vertices.size == 0:

        return mesh

    z_values = vertices[:, 2]
    z_min = float(z_values.min())
    z_max = float(z_values.max())
    anchor = (z_min + z_max) * 0.5

    if z_max <= z_min:

        return mesh

    back_mask = z_values < anchor

    if not np.any(back_mask):

        return mesh

    back_span = max(
        anchor - z_min,
        1e-6
    )

    distance = (
        anchor - z_values[back_mask]
    ) / back_span

    local_strength = strength * np.clip(
        distance,
        0.0,
        1.0
    ) ** 0.75

    vertices[back_mask, 2] = (
        z_values[back_mask]
        + (anchor - z_values[back_mask]) * local_strength
    )

    mesh.vertices = vertices

    try:

        mesh.fix_normals()

    except Exception:

        pass

    print(
        "Back depth constrained:",
        f"{z_max - z_min:.3f}",
        "->",
        f"{float(vertices[:, 2].max() - vertices[:, 2].min()):.3f}"
    )

    return mesh


def clean_mesh(mesh, smooth_iterations: int = 0):

    mesh = keep_primary_components(
        mesh
    )

    try:

        mesh.remove_degenerate_faces()

    except Exception:

        pass

    mesh.remove_unreferenced_vertices()

    try:

        mesh.merge_vertices()

    except Exception:

        pass

    mesh = smooth_mesh(
        mesh,
        smooth_iterations
    )

    mesh.remove_unreferenced_vertices()

    try:

        mesh.fix_normals()

    except Exception:

        pass

    return mesh

# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return FileResponse("index.html")

# =====================================================
# PROGRESS
# =====================================================

@app.get("/progress")
def progress():

    return JSONResponse(progress_data)

# =====================================================
# CLEAR OUTPUTS
# =====================================================

@app.post("/clear_outputs")
def clear_outputs():

    try:

        for item in os.listdir(OUTPUT_DIR):

            item_path = os.path.join(
                OUTPUT_DIR,
                item
            )

            try:

                if os.path.isfile(item_path):

                    os.remove(item_path)

                elif os.path.isdir(item_path):

                    shutil.rmtree(item_path)

            except Exception as e:

                print("Delete failed:", e)

        return {
            "success": True
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =====================================================
# GENERATE 3D
# =====================================================

@app.get("/generate3d")
def generate3d(
    prompt: str,
    seed: Optional[int] = None,
    source: str = "sdxl",
    quality: str = DEFAULT_MESH_QUALITY,
    mc_resolution: Optional[int] = None
):

    try:

        progress_data["progress"] = 5
        progress_data["status"] = "Preparing job..."

        uid = str(uuid.uuid4())[:8]

        out_dir = os.path.join(
            OUTPUT_DIR,
            uid
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        raw_path = os.path.join(
            out_dir,
            "raw.png"
        )

        input_path = os.path.join(
            out_dir,
            "input.png"
        )

        glb_path = os.path.join(
            out_dir,
            "mesh.glb"
        )

        mesh_settings = resolve_mesh_settings(
            quality,
            mc_resolution
        )

        if source == "example":

            progress_data["progress"] = 15
            progress_data["status"] = "Loading example image..."

            image = load_example_image()

            seed_used = seed

        else:

            progress_data["progress"] = 15
            progress_data["status"] = "Generating source image..."

            image, seed_used = generate_source_image(
                prompt,
                seed
            )

        image.save(raw_path)

        progress_data["progress"] = 40
        progress_data["status"] = "Isolating object..."

        image = preprocess_for_triposr(image)

        image.save(input_path)

        progress_data["progress"] = 55
        progress_data["status"] = "Running TripoSR..."

        with torch.no_grad():

            scene_codes = triposr_model(
                [image],
                device=device
            )

        progress_data["progress"] = 78
        progress_data["status"] = "Extracting mesh..."

        meshes = triposr_model.extract_mesh(
            scene_codes,
            True,
            resolution=mesh_settings["resolution"],
            threshold=mesh_settings["threshold"]
        )

        mesh = clean_mesh(
            meshes[0],
            smooth_iterations=mesh_settings["smooth_iterations"]
        )

        mesh = to_gradio_3d_orientation(
            mesh
        )

        mesh = constrain_unseen_back_depth(
            mesh,
            prompt
        )

        progress_data["progress"] = 92
        progress_data["status"] = "Exporting GLB..."

        mesh.export(glb_path)

        if not os.path.exists(glb_path):

            raise Exception(
                "GLB export failed"
            )

        if os.path.getsize(glb_path) == 0:

            raise Exception(
                "GLB file empty"
            )

        print("GLB SAVED:", glb_path)

        progress_data["progress"] = 100
        progress_data["status"] = "Done"

        return {
            "model_url":
                f"/outputs/{uid}/mesh.glb",

            "image_url":
                f"/outputs/{uid}/input.png",

            "raw_image_url":
                f"/outputs/{uid}/raw.png",

            "seed":
                seed_used,

            "mc_resolution":
                mesh_settings["resolution"],

            "mesh_quality":
                mesh_settings["quality"],

            "smooth_iterations":
                mesh_settings["smooth_iterations"],

            "mesh_threshold":
                mesh_settings["threshold"],

            "back_depth_constraint":
                MESH_BACK_DEPTH_CONSTRAINT_ENABLED,

            "image_settings": {
                "model_kind": IMAGE_MODEL_KIND,
                "model_id": SDXL_MODEL_ID if IMAGE_MODEL_KIND == "sdxl" else SD15_MODEL_ID,
                "width": SDXL_WIDTH,
                "height": SDXL_HEIGHT,
                "steps": SDXL_STEPS,
                "guidance_scale": SDXL_GUIDANCE_SCALE
            }
        }

    except Exception as e:

        print("ERROR:", e)

        progress_data["progress"] = 0

        progress_data["status"] = str(e)

        return {
            "error": str(e)
        }

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
