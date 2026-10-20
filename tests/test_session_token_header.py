import io

from fastapi.testclient import TestClient
from openspline_server.app import create_app
from openspline_server.config import Settings
from PIL import Image


def png(color="blue"):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


def test_session_token_header_authenticates_beside_basic_auth(tmp_path):
    """A proxy with HTTP basic auth hides the caller's Authorization header."""
    app = create_app(Settings(backend="test", runtime_dir=str(tmp_path)))
    with TestClient(app) as client:
        response = client.post("/v1/sessions", files={"portrait": ("p.png", png())})
        assert response.status_code == 201, response.text
        descriptor = response.json()
        url = f"/v1/sessions/{descriptor['id']}"

        # The browser replayed Basic credentials instead of the Bearer token.
        basic = client.get(
            url + "/portrait", headers={"Authorization": "Basic YmFuZTpwYXNz"}
        )
        assert basic.status_code == 401

        # The dedicated header carries the token past the proxy.
        poster = client.get(
            url + "/portrait", headers={"X-Session-Token": descriptor["token"]}
        )
        assert poster.status_code == 200
        assert Image.open(io.BytesIO(poster.content)).size == (512, 512)

        assert (
            client.delete(
                url, headers={"X-Session-Token": descriptor["publisher_token"]}
            ).status_code
            == 204
        )
