import argparse
import logging
import os
import re
import time
import uuid

import numpy as np
import rembg
import torch
import trimesh
import xatlas

from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from diffusers import StableDiffusionXLPipeline

try:
    from diffusers import DPMSolverMultistepScheduler
except ImportError:
    DPMSolverMultistepScheduler = None

from tsr.system import TSR
from tsr.utils import (
    remove_background,
    resize_foreground,
)

from tsr.bake_texture import bake_texture


# =====================================================
# TIMER
# =====================================================

class Timer:

    def __init__(self):

        self.items = {}

        self.time_scale = 1000.0

        self.time_unit = "ms"

    def start(self, name: str):

        if torch.cuda.is_available():

            torch.cuda.synchronize()

        self.items[name] = time.time()

        logging.info(f"{name} ...")

    def end(self, name: str):

        if name not in self.items:

            return

        if torch.cuda.is_available():

            torch.cuda.synchronize()

        start_time = self.items.pop(name)

        delta = time.time() - start_time

        t = delta * self.time_scale

        logging.info(
            f"{name} finished in {t:.2f}{self.time_unit}."
        )


timer = Timer()


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# =====================================================
# DEVICE
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

DEFAULT_MESH_THRESHOLD = float(
    os.getenv("TRIPOSR_ISOSURFACE_THRESHOLD", "28.0")
)

MESH_BACK_DEPTH_CONSTRAINT_ENABLED = os.getenv(
    "TRIPOSR_BACK_DEPTH_CONSTRAINT",
    "1"
) != "0"

MESH_BACK_DEPTH_STRENGTH = float(
    os.getenv("TRIPOSR_BACK_DEPTH_STRENGTH", "0.45")
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# =====================================================
# LOAD SDXL
# =====================================================

print("Loading SDXL...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=(
        torch.float16
        if device == "cuda"
        else torch.float32
    )
)

if DPMSolverMultistepScheduler is not None:

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

pipe = pipe.to(device)

pipe.enable_attention_slicing()

pipe.safety_checker = None

print("SDXL loaded")


# =====================================================
# LOAD TRIPOSR
# =====================================================

timer.start("Initializing model")

model = TSR.from_pretrained(
    "./TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

model.renderer.set_chunk_size(8192)

model.to(device)

timer.end("Initializing model")


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
    "background",
    "environment",
    "room",
    "floor",
    "table",
    "ground",
    "wall",
    "scene",
    "shadow",
    "reflection",
    "multiple objects",
    "duplicate object",
    "second object",
    "object pair",
    "two views",
    "reference sheet",
    "inset image",
    "split screen",
    "comparison image",
    "cropped",
    "photo",
    "photography",
    "realistic scene",
    "complex background",
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


def build_prompt(user_prompt):

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

    single isolated object,
    centered object,
    clean silhouette,
    {emphasis}
    front facing,
    game asset,
    smooth clean surfaces,
    simple geometry,
    crisp object boundaries,
    solid material,
    matte product render,
    no environment,
    no room,
    no floor,
    no shadows,
    no reflections,
    studio render
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
# MESH HELPERS
# =====================================================

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


def smooth_mesh(mesh, iterations: int = 8):

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

    depth, width, height = extents[:3]

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

    x_values = vertices[:, 0]
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    anchor = (x_min + x_max) * 0.5

    if x_max <= x_min:

        return mesh

    back_mask = x_values > anchor

    if not np.any(back_mask):

        return mesh

    back_span = max(
        x_max - anchor,
        1e-6
    )

    distance = (
        x_values[back_mask] - anchor
    ) / back_span

    local_strength = strength * np.clip(
        distance,
        0.0,
        1.0
    ) ** 0.75

    vertices[back_mask, 0] = (
        x_values[back_mask]
        - (x_values[back_mask] - anchor) * local_strength
    )

    mesh.vertices = vertices

    try:

        mesh.fix_normals()

    except Exception:

        pass

    logging.info(
        "Back depth constrained: %.3f -> %.3f",
        x_max - x_min,
        float(vertices[:, 0].max() - vertices[:, 0].min())
    )

    return mesh


def clean_mesh(mesh, smooth_iterations: int = 8):

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
# IMAGE HELPERS
# =====================================================

def keep_main_foreground_object(image: Image.Image) -> Image.Image:

    if cv2 is None or image.mode != "RGBA":

        return image

    arr = np.array(
        image
    )

    mask = (
        arr[:, :, 3] > 10
    ).astype(np.uint8)

    if mask.max() == 0:

        return image

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        8
    )

    if count <= 2:

        return image

    largest_label = int(
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
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


# =====================================================
# GENERATE
# =====================================================

def generate(prompt):

    uid = str(uuid.uuid4())[:8]

    out_dir = os.path.join(
        OUTPUT_DIR,
        uid
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    image_path = os.path.join(
        out_dir,
        "input.png"
    )

    glb_path = os.path.join(
        out_dir,
        "mesh.glb"
    )

    # =================================================
    # GENERATE IMAGE
    # =================================================

    timer.start("Generating image")

    result = pipe(
        prompt=build_prompt(prompt),
        negative_prompt=build_negative_prompt(prompt),
        num_inference_steps=30,
        guidance_scale=7,
        width=1024,
        height=1024
    )

    image = result.images[0]

    timer.end("Generating image")

    # =================================================
    # OFFICIAL TRIPOSR PREPROCESSING
    # =================================================

    timer.start("Processing images")

    rembg_session = rembg.new_session()

    image = remove_background(
        image,
        rembg_session
    )

    image = keep_main_foreground_object(
        image
    )

    image = resize_foreground(
        image,
        0.85
    )

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

    image = Image.fromarray(
        (image * 255.0).astype(np.uint8)
    )

    image.save(image_path)

    timer.end("Processing images")

    # =================================================
    # RUN MODEL
    # =================================================

    timer.start("Running model")

    with torch.no_grad():

        scene_codes = model(
            [image],
            device=device
        )

    timer.end("Running model")

    # =================================================
    # EXTRACT MESH
    # =================================================

    timer.start("Extracting mesh")

    meshes = model.extract_mesh(
        scene_codes,
        True,
        resolution=256,
        threshold=DEFAULT_MESH_THRESHOLD
    )

    mesh = clean_mesh(
        meshes[0]
    )

    mesh = constrain_unseen_back_depth(
        mesh,
        prompt
    )

    timer.end("Extracting mesh")

    # =================================================
    # EXPORT
    # =================================================

    timer.start("Exporting mesh")

    mesh.export(glb_path)

    timer.end("Exporting mesh")

    return {
        "model_url":
            f"/outputs/{uid}/mesh.glb",

        "image_url":
            f"/outputs/{uid}/input.png"
    }


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="smooth toy robot"
    )

    args = parser.parse_args()

    result = generate(args.prompt)

    print(result)
