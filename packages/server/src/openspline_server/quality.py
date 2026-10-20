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


def profile_for(quality, devices=1, fps=None, size=None):
    """Resolve the model, output size and frame rate for a quality tier.

    The tiers keep the frame rates their models were trained at. Lowering the
    Pro model's rate makes it animate the audio too slowly, so the lip sync
    drifts; shrinking its output instead changes the framing the model was
    trained with. Neither is a good trade, so a tier only changes when the
    deployment asks for it explicitly.
    """
    if quality not in QUALITY_PROFILES:
        raise ValueError("quality must be " + ", ".join(QUALITY_PROFILES))
    profile = dict(QUALITY_PROFILES[quality])
    if size is not None:
        profile["size"] = int(size)
    if fps is not None:
        profile["fps"] = int(fps)
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
