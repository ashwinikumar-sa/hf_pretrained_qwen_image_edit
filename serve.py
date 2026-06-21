"""
FastAPI inference server for Qwen-Image-Edit on Trainium2.

Models are loaded ONCE at startup and reused across all requests.

Usage:
    NEURON_RT_NUM_CORES=32 python serve.py \
        --compiled_models_dir /opt/dlami/nvme/compiled_models_tp16 \
        --height 1024 --width 512 \
        --use_v3_cfg --patch_multiplier 3 \
        --port 8080

Request (multipart/form-data):
    POST /infer
      - image1: file           (required - garment image)
      - image2: file           (optional - person/model image)
      - prompt: str            (required)
      - negative_prompt: str   (optional)
      - num_inference_steps: int  (optional, default from CLI)
      - true_cfg_scale: float  (optional, default from CLI)
      - seed: int              (optional, default 42)

Response:
    image/png  (edited output image)
    Header: X-Inference-Time  (seconds)
"""

import os
import argparse

# ── Neuron env vars must be set before any torch/neuron imports ──────────────
WORLD_SIZE = int(os.environ.get("NEURON_RT_NUM_CORES", "8"))
os.environ["LOCAL_WORLD_SIZE"] = str(WORLD_SIZE)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
os.environ["NEURON_CUSTOM_SILU"] = "1"

# Parse server args early so they're available at module level
_parser = argparse.ArgumentParser(description="Qwen-Image-Edit inference server")
_parser.add_argument("--host", type=str, default="127.0.0.1")
_parser.add_argument("--port", type=int, default=8080)
_parser.add_argument("--compiled_models_dir", type=str,
                     default="/opt/dlami/nvme/compiled_models_tp16")
_parser.add_argument("--height", type=int, default=1024)
_parser.add_argument("--width", type=int, default=512)
_parser.add_argument("--patch_multiplier", type=int, default=3)
_parser.add_argument("--num_inference_steps", type=int, default=50)
_parser.add_argument("--true_cfg_scale", type=float, default=3.0)
_parser.add_argument("--use_v3_cfg", action="store_true", default=True)
SERVER_ARGS = _parser.parse_args()

import io
import random
import tempfile
import time
import types

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Security, Request
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image

from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
import run_qwen_image_edit as inference_module

app = FastAPI(
    title="Qwen-Image-Edit Inference Server",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_security = HTTPBearer(auto_error=False)

def _verify_token(credentials: HTTPAuthorizationCredentials = Security(_security)):
    expected = os.environ.get("API_SECRET_TOKEN")
    # Return identical opaque 403 regardless of whether path exists or token is wrong
    # so an attacker cannot distinguish "wrong token" from "path doesn't exist"
    if not expected or credentials is None or credentials.credentials != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

# Global: pipeline loaded once at startup
_pipe = None


def _base_args() -> types.SimpleNamespace:
    """Build a base args Namespace with server-level defaults."""
    return types.SimpleNamespace(
        images=[],
        prompt="",
        negative_prompt="",
        output=None,
        height=SERVER_ARGS.height,
        width=SERVER_ARGS.width,
        patch_multiplier=SERVER_ARGS.patch_multiplier,
        image_size=448,
        max_sequence_length=1024,
        vision_tp=False,
        cpu_language_model=True,
        neuron_language_model=False,
        use_v3_language_model=True,
        cpu_vision_encoder=False,
        neuron_vision_encoder=False,
        use_v3_vision_encoder=True,
        num_inference_steps=SERVER_ARGS.num_inference_steps,
        true_cfg_scale=SERVER_ARGS.true_cfg_scale,
        seed=42,
        compiled_models_dir=SERVER_ARGS.compiled_models_dir,
        vae_tile_size=512,
        use_v2=False,
        use_v1_flash=False,
        use_v2_flash=False,
        use_v3_cp=False,
        use_v3_cfg=SERVER_ARGS.use_v3_cfg,
        warmup=False,
        save_comparison=False,
        cpu_vae_decode=False,
        debug_text_encoder=False,
    )


@app.on_event("startup")
def load_pipeline():
    """Load the pipeline and all compiled Neuron models once at startup."""
    global _pipe

    args = _base_args()
    dtype = torch.bfloat16

    print("\nLoading base pipeline...")

    # Override VAE_IMAGE_SIZE to match compiled dimensions (same as run_inference does)
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_pipeline_module
    compiled_vae_pixels = args.height * args.width
    qwen_pipeline_module.VAE_IMAGE_SIZE = compiled_vae_pixels

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        inference_module.MODEL_ID,
        torch_dtype=dtype,
        cache_dir=inference_module.HUGGINGFACE_CACHE_DIR,
        local_files_only=True,
    )

    # Fix processor pixel constraints to match compiled vision encoder
    target_pixels = args.image_size * args.image_size
    pipe.processor.image_processor.min_pixels = target_pixels
    pipe.processor.image_processor.max_pixels = target_pixels

    print("Loading compiled Neuron models (this takes a few minutes)...")
    pipe = inference_module.load_all_compiled_models(args.compiled_models_dir, pipe, args)

    _pipe = pipe
    print("\nPipeline ready. Server accepting requests.")


@app.get("/health")
def health(_: None = Depends(_verify_token)):
    return {"status": "ok" if _pipe is not None else "loading", "world_size": WORLD_SIZE}


@app.post("/infer")
async def infer(
    image1: UploadFile = File(..., description="First input image (e.g. garment)"),
    image2: UploadFile = File(None, description="Second input image (e.g. person)"),
    prompt: str = Form(...),
    negative_prompt: str = Form("blurry, low quality, deformed, distorted"),
    num_inference_steps: int = Form(SERVER_ARGS.num_inference_steps),
    true_cfg_scale: float = Form(SERVER_ARGS.true_cfg_scale),
    seed: int = Form(42),
    _: None = Depends(_verify_token),
):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready yet")

    start = time.time()
    tmp_files = []

    try:
        # Save uploads to temp files and resize to compiled dimensions
        def save_and_resize(data: bytes) -> str:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img = img.resize((SERVER_ARGS.width, SERVER_ARGS.height))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f, format="PNG")
                return f.name

        tmp_files.append(save_and_resize(await image1.read()))
        if image2 is not None:
            tmp_files.append(save_and_resize(await image2.read()))

        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        # Load PIL images from temp files
        source_images = [load_image(p) for p in tmp_files]
        input_images = source_images[0] if len(source_images) == 1 else source_images

        try:
            output = _pipe(
                image=input_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=SERVER_ARGS.height,
                width=SERVER_ARGS.width,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        buf = io.BytesIO()
        output.images[0].save(buf, format="PNG")
        buf.seek(0)

        elapsed = time.time() - start
        print(f"Inference done in {elapsed:.1f}s | steps={num_inference_steps} cfg={true_cfg_scale} seed={seed}")

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"X-Inference-Time": f"{elapsed:.2f}s"},
        )

    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_ARGS.host, port=SERVER_ARGS.port, server_header=False)


# Catch-all — must be registered LAST so real routes take priority.
# Any undefined path returns the same opaque 403 as a bad token,
# so an attacker cannot determine what paths exist.
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def catch_all(path: str, request: Request):
    raise HTTPException(status_code=403, detail="Forbidden")
