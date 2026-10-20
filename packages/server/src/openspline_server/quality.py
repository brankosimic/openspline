"""Local model checks for demo quality switching; never download during playback."""

import json
from pathlib import Path

# Quality tiers. "low" is the distilled Lite model at 512; "medium" is the same
# model at 768 for more spatial detail; "high" is the Pro model, which costs
# roughly seven times more per frame because its VAE keeps four times the
# spatial resolution.
QUALITY_PROFILES = {
    "low": {"model": "lite", "size": 512, "fps": 25},
    "medium": {"model": "lite", "size": 768, "fps": 25},
    "high": {"model": "pro", "size": 512, "fps": 25},
}
QUALITIES = tuple(QUALITY_PROFILES)
# The Pro model cannot sustain 25 fps on one GPU, so it runs slower there and
# the playout timeline follows the model's rate instead of starving. Eight
# frames per second is the break-even for the Pro model on a single 24 GB GPU:
# it produces a little faster than real time, so the viewer never runs dry.
SINGLE_GPU_PRO_FPS = 8


def profile_for(quality, devices=1, fps=None):
    """Resolve the model, output size and frame rate for a quality tier."""
    if quality not in QUALITY_PROFILES:
        raise ValueError("quality must be " + ", ".join(QUALITY_PROFILES))
    profile = dict(QUALITY_PROFILES[quality])
    if fps is not None:
        profile["fps"] = int(fps)
    elif profile["model"] == "pro" and devices < 2:
        profile["fps"] = SINGLE_GPU_PRO_FPS
    return profile


def _weights_present(directory, names):
    for name in names:
        path = directory / name
        if path.is_file() and path.stat().st_size:
            return True
        index = directory / (name + ".index.json")
        if index.is_file():
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
                if shards and all(
                    (directory / shard).is_file() and (directory / shard).stat().st_size
                    for shard in shards
                ):
                    return True
            except (ValueError, KeyError, TypeError, OSError):
                pass
    return False


async def prepare_quality(settings, quality):
    if settings.backend == "test":
        return
    avatar, audio = Path(settings.model_dir), Path(settings.audio_model_dir)
    model = avatar / ("Model_Lite" if QUALITY_PROFILES[quality]["model"] == "lite" else "Model_Pro")
    ready = (model / "config.json").is_file() and _weights_present(
        model, ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"]
    )
    if QUALITY_PROFILES[quality]["model"] == "lite":
        vae = avatar / "VAE_LTX"
        ready = (
            ready
            and (vae / "config.json").is_file()
            and _weights_present(vae, ["autoencoder.pth", "diffusion_pytorch_model.safetensors"])
        )
    else:
        ready = ready and _weights_present(avatar / "VAE_Wan", ["Wan2.1_VAE.pth"])
    ready = ready and all(
        (audio / name).is_file() for name in ["config.json", "preprocessor_config.json"]
    )
    ready = ready and _weights_present(audio, ["model.safetensors", "pytorch_model.bin"])
    if not ready:
        raise ValueError(
            f"{quality.capitalize()}-quality model files are missing or incomplete. "
            "Stop the service and rerun the installer or run.sh --prepare before restarting. "
            "The demo does not download models."
        )
