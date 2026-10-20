import io

import pytest
from fastapi.testclient import TestClient
from openspline_server.app import create_app
from openspline_server.config import Settings, WorkerConfig
from openspline_server.quality import QUALITIES, profile_for
from PIL import Image


def portrait():
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "blue").save(output, format="PNG")
    return output.getvalue()


def test_quality_profiles_pick_a_model_size_and_frame_rate():
    assert QUALITIES == ("low", "medium", "high")
    assert profile_for("low") == {"model": "lite", "size": 512, "fps": 25}
    assert profile_for("medium") == {"model": "lite", "size": 768, "fps": 25}
    # One GPU renders the Pro model smaller so it keeps the 25 FPS cadence.
    assert profile_for("high", devices=1) == {"model": "pro", "size": 320, "fps": 25}
    assert profile_for("high", devices=2) == {"model": "pro", "size": 512, "fps": 25}
    # Explicit frame rate and size always win.
    assert profile_for("high", devices=1, fps=20)["fps"] == 20
    assert profile_for("high", devices=1, size=384)["size"] == 384
    with pytest.raises(ValueError):
        profile_for("ultra")


def test_worker_config_accepts_the_new_tier_and_validates_the_frame_rate():
    settings = Settings(workers=[WorkerConfig("a", "medium", (0,))])
    assert settings.workers[0].quality == "medium"
    with pytest.raises(ValueError):
        Settings(workers=[WorkerConfig("a", "ultra", (0,))])
    with pytest.raises(ValueError):
        Settings(workers=[WorkerConfig("a", "low", (0,), fps=0)])
    with pytest.raises(ValueError):
        Settings(workers=[WorkerConfig("a", "medium", (0, 1))])
    with pytest.raises(ValueError):
        Settings(workers=[WorkerConfig("a", "low", (0,), size=300)])


def test_session_descriptor_reports_the_frame_size(tmp_path):
    settings = Settings(
        backend="test", runtime_dir=str(tmp_path), workers=[WorkerConfig("a", "medium", (0,))]
    )
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions", data={"quality": "medium"}, files={"portrait": ("p.png", portrait())}
        )
        assert response.status_code == 201, response.text
        descriptor = response.json()
        assert descriptor["size"] == 512  # the test backend renders at 512
        assert descriptor["fps"] == 25
        client.delete(
            f"/v1/sessions/{descriptor['id']}",
            headers={"Authorization": "Bearer " + descriptor["publisher_token"]},
        )
