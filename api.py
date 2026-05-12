import os
import re
import shutil
import sys
import uuid
import math
from typing import Optional

import numpy as np
import rembg
import torch
import trimesh
import uvicorn

from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPOSR_DIR = os.path.join(BASE_DIR, "TripoSR")

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

if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

try:
    from tsr.bake_texture import bake_texture
except Exception:
    bake_texture = None

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
        "candidate_views": 1,
        "texture_resolution": 0,
        "target_faces": 45000,
        "advanced_cleanup": False,
    },
    "balanced": {
        "resolution": DEFAULT_MC_RESOLUTION,
        "smooth_iterations": 8,
        "threshold": DEFAULT_MESH_THRESHOLD,
        "candidate_views": 1,
        "texture_resolution": 0,
        "target_faces": 65000,
        "advanced_cleanup": True,
    },
    "high": {
        "resolution": 320,
        "smooth_iterations": 6,
        "threshold": DEFAULT_MESH_THRESHOLD,
        "candidate_views": 4,
        "texture_resolution": 1024,
        "target_faces": 90000,
        "advanced_cleanup": True,
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

HIGH_DETAIL_CANDIDATE_VIEWS = [
    "front three-quarter orthographic product view, all parts visible",
    "front orthographic product view, symmetrical full object",
    "left side orthographic product view, full object silhouette",
    "rear three-quarter orthographic product view, full object visible",
]

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
        "recognizable table structure, flat rectangular tabletop, straight supporting legs, hard clean edges, visible underside supports, no extra duplicate table",
    ),
    (
        ["fan"],
        "recognizable fan structure, circular guard, central motor hub, visible blades, straight pole, stable base, symmetrical round parts",
    ),
    (
        ["car", "vehicle", "truck", "bus"],
        "recognizable vehicle structure, four wheels, clean body panels, hard surface edges, symmetrical left and right sides",
    ),
    (
        ["nok", "terracotta", "artefact", "artifact"],
        "archaeological terracotta artifact, weathered clay material, carved relief details, chipped irregular edges, thick solid fragment, museum scan style",
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


HARD_SURFACE_TERMS = [
    "table",
    "desk",
    "chair",
    "stool",
    "fan",
    "vehicle",
    "car",
    "truck",
    "bus",
    "box",
    "cabinet",
    "shelf",
]


FURNITURE_TERMS = [
    "table",
    "desk",
    "chair",
    "stool",
    "bench",
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


def prompt_has_any(prompt: str, terms: list[str]) -> bool:

    return any(
        prompt_has_term(prompt, term)
        for term in terms
    )


def build_prompt(user_prompt: str, view_hint: Optional[str] = None) -> str:

    subject = user_prompt.strip()

    subject = re.sub(
        r"\s+",
        " ",
        subject
    )

    subject_parts = [
        part.strip()
        for part in subject.split(",")
        if part.strip()
    ]

    if len(subject_parts) > 6:

        subject = ", ".join(
            subject_parts[:6]
        )

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

    view_text = view_hint or "front three-quarter orthographic product view"

    prompt_parts = [
        subject,
        "single complete object",
        "centered full object visible",
        view_text,
        "clean readable silhouette",
    ]

    prompt_parts.extend(identity_rules[:1])
    prompt_parts.extend(style_rules[:1])

    prompt_parts.extend(
        [
            "accurate proportions",
            "crisp object boundaries",
            "matte product render",
            "neutral gray studio background",
            "sharp focus",
        ]
    )

    return ", ".join(
        part
        for part in prompt_parts
        if part
    )


def build_negative_prompt(user_prompt: str) -> str:

    negative_terms = [
        term
        for term in BASE_NEGATIVE_TERMS
        if not prompt_has_term(user_prompt, term)
    ]

    return ", ".join(
        negative_terms[:24]
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


def generate_source_image(
    prompt: str,
    seed: Optional[int],
    view_hint: Optional[str] = None
) -> tuple[Image.Image, int]:

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
        prompt=build_prompt(prompt, view_hint),
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


def repair_mesh(mesh):

    try:

        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

    except Exception as e:

        print("Mesh repair skipped:", e)

    try:

        mesh.remove_degenerate_faces()

    except Exception:

        pass

    mesh.remove_unreferenced_vertices()

    try:

        mesh.merge_vertices()

    except Exception:

        pass

    return mesh


def decimate_mesh(mesh, target_faces: int):

    if target_faces <= 0 or len(mesh.faces) <= target_faces:

        return mesh

    try:

        target_reduction = 1.0 - (
            float(target_faces) / float(len(mesh.faces))
        )

        target_reduction = max(
            0.0,
            min(
                target_reduction,
                0.95
            )
        )

        return mesh.simplify_quadric_decimation(
            percent=target_reduction
        )

    except Exception as e:

        try:

            return mesh.simplify_quadric_decimation(
                face_count=int(target_faces)
            )

        except Exception as retry_error:

            print("Decimation skipped:", retry_error or e)

    return mesh


def normalize_mesh_transform(mesh):

    bounds = np.asarray(
        mesh.bounds,
        dtype=np.float64
    )

    if bounds.shape != (2, 3):

        return mesh

    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    max_extent = float(extents.max())

    if max_extent <= 0:

        return mesh

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64
    ).copy()

    vertices = (vertices - center) / max_extent * 2.0

    vertices[:, 1] -= vertices[:, 1].min()

    mesh.vertices = vertices

    try:

        mesh.fix_normals()

    except Exception:

        pass

    return mesh


def snap_near_ground_for_furniture(mesh, prompt: str):

    if not prompt_has_any(prompt, FURNITURE_TERMS):

        return mesh

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64
    ).copy()

    if vertices.size == 0:

        return mesh

    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    height = max(y_max - y_min, 1e-6)

    lower_band = vertices[:, 1] <= y_min + height * 0.08

    if np.any(lower_band):

        vertices[lower_band, 1] = y_min
        mesh.vertices = vertices

        try:

            mesh.fix_normals()

        except Exception:

            pass

    return mesh


def preserve_hard_surface_shape(mesh, prompt: str):

    if not prompt_has_any(prompt, HARD_SURFACE_TERMS):

        return mesh

    try:

        mesh.fix_normals()

    except Exception:

        pass

    return mesh


def score_mesh_candidate(mesh, prompt: str) -> float:

    try:

        extents = np.asarray(
            mesh.extents,
            dtype=np.float64
        )

        face_count = len(mesh.faces)
        vertex_count = len(mesh.vertices)
        component_count = len(mesh.split(only_watertight=False))
        volume_score = float(np.prod(np.maximum(extents, 1e-6)))
        compactness = float(extents.min() / max(extents.max(), 1e-6))

        score = (
            math.log1p(face_count)
            + math.log1p(vertex_count) * 0.5
            + volume_score * 0.15
            + compactness
            - max(0, component_count - 1) * 0.75
        )

        if prompt_has_any(prompt, FURNITURE_TERMS):

            score += float(extents[0] + extents[2]) * 0.2

        return score

    except Exception:

        return 0.0


def advanced_mesh_cleanup(mesh, prompt: str, mesh_settings: dict):

    mesh = repair_mesh(mesh)

    mesh = snap_near_ground_for_furniture(
        mesh,
        prompt
    )

    mesh = preserve_hard_surface_shape(
        mesh,
        prompt
    )

    mesh = decimate_mesh(
        mesh,
        int(mesh_settings.get("target_faces") or 0)
    )

    mesh = repair_mesh(mesh)

    mesh = normalize_mesh_transform(mesh)

    return mesh


def bake_mesh_texture(mesh, scene_code, texture_resolution: int, texture_path: str):

    if texture_resolution <= 0 or bake_texture is None:

        return mesh, False

    try:

        bake_output = bake_texture(
            mesh,
            triposr_model,
            scene_code,
            texture_resolution
        )

        texture_image = Image.fromarray(
            (bake_output["colors"] * 255.0)
            .clip(0, 255)
            .astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)

        texture_image.save(texture_path)

        visual = trimesh.visual.TextureVisuals(
            uv=bake_output["uvs"],
            image=texture_image
        )

        textured_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[bake_output["vmapping"]],
            faces=bake_output["indices"],
            vertex_normals=mesh.vertex_normals[bake_output["vmapping"]],
            visual=visual,
            process=False
        )

        return textured_mesh, True

    except Exception as e:

        print("Texture baking skipped:", e)

    return mesh, False

# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return FileResponse("index.html")


@app.get("/health")
def health():

    return {
        "ok": True
    }

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

        texture_path = os.path.join(
            out_dir,
            "texture.png"
        )

        glb_path = os.path.join(
            out_dir,
            "mesh.glb"
        )

        mesh_settings = resolve_mesh_settings(
            quality,
            mc_resolution
        )

        candidate_views = HIGH_DETAIL_CANDIDATE_VIEWS[
            :max(1, int(mesh_settings.get("candidate_views") or 1))
        ]

        if source == "example":

            progress_data["progress"] = 15
            progress_data["status"] = "Loading example image..."

            image = load_example_image()

            seed_used = seed

            candidate_images = [
                {
                    "image": image,
                    "seed": seed_used,
                    "view": "example image"
                }
            ]

        else:

            progress_data["progress"] = 15
            progress_data["status"] = "Generating source image candidates..."

            candidate_images = []

            base_seed = seed

            for index, view_hint in enumerate(candidate_views):

                candidate_seed = None

                if base_seed is not None:

                    candidate_seed = int(base_seed) + index

                image, seed_used = generate_source_image(
                    prompt,
                    candidate_seed,
                    view_hint
                )

                candidate_images.append(
                    {
                        "image": image,
                        "seed": seed_used,
                        "view": view_hint
                    }
                )

                progress_data["progress"] = 15 + int(
                    (index + 1) / len(candidate_views) * 20
                )

        best_candidate = None
        best_score = -float("inf")

        for index, candidate in enumerate(candidate_images):

            candidate_raw_path = os.path.join(
                out_dir,
                f"raw_{index + 1}.png"
            )

            candidate_input_path = os.path.join(
                out_dir,
                f"input_{index + 1}.png"
            )

            candidate["image"].save(candidate_raw_path)

            progress_data["progress"] = 40 + int(
                index / len(candidate_images) * 35
            )
            progress_data["status"] = (
                f"Reconstructing candidate {index + 1}/{len(candidate_images)}..."
            )

            processed_image = preprocess_for_triposr(
                candidate["image"]
            )

            processed_image.save(candidate_input_path)

            with torch.no_grad():

                scene_codes = triposr_model(
                    [processed_image],
                    device=device
                )

            meshes = triposr_model.extract_mesh(
                scene_codes,
                True,
                resolution=mesh_settings["resolution"],
                threshold=mesh_settings["threshold"]
            )

            smooth_iterations = mesh_settings["smooth_iterations"]

            if prompt_has_any(prompt, HARD_SURFACE_TERMS):

                smooth_iterations = min(
                    smooth_iterations,
                    2
                )

            original_mesh = clean_mesh(
                meshes[0],
                smooth_iterations=smooth_iterations
            )

            mesh = to_gradio_3d_orientation(
                original_mesh.copy()
            )

            mesh = constrain_unseen_back_depth(
                mesh,
                prompt
            )

            if mesh_settings.get("advanced_cleanup"):

                mesh = advanced_mesh_cleanup(
                    mesh,
                    prompt,
                    mesh_settings
                )

            else:

                mesh = normalize_mesh_transform(mesh)

            score = score_mesh_candidate(
                mesh,
                prompt
            )

            if score > best_score:

                best_score = score

                best_candidate = {
                    "mesh": mesh,
                    "original_mesh": original_mesh,
                    "scene_code": scene_codes[0],
                    "raw_path": candidate_raw_path,
                    "input_path": candidate_input_path,
                    "seed": candidate["seed"],
                    "view": candidate["view"],
                    "score": score,
                }

        if best_candidate is None:

            raise Exception(
                "No mesh candidate was generated"
            )

        shutil.copyfile(
            best_candidate["raw_path"],
            raw_path
        )

        shutil.copyfile(
            best_candidate["input_path"],
            input_path
        )

        seed_used = best_candidate["seed"]

        mesh = best_candidate["mesh"]

        texture_baked = False

        if mesh_settings["texture_resolution"] > 0:

            progress_data["progress"] = 84
            progress_data["status"] = "Baking texture atlas..."

            textured_mesh, texture_baked = bake_mesh_texture(
                best_candidate["original_mesh"],
                best_candidate["scene_code"],
                int(mesh_settings["texture_resolution"]),
                texture_path
            )

            if texture_baked:

                mesh = to_gradio_3d_orientation(
                    textured_mesh
                )

                mesh = constrain_unseen_back_depth(
                    mesh,
                    prompt
                )

                mesh = normalize_mesh_transform(mesh)

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

            "candidate_views":
                len(candidate_images),

            "selected_view":
                best_candidate["view"],

            "candidate_score":
                best_candidate["score"],

            "texture_baked":
                texture_baked,

            "texture_resolution":
                mesh_settings["texture_resolution"],

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
