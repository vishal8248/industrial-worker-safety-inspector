import json
from pathlib import Path


class ZoneConfig:
    def __init__(
        self,
        config_path: str = "config/camera_zones.json",
    ):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Zone config not found: {self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

    def get_camera_config(
        self,
        camera_id: str,
    ) -> dict:
        if camera_id not in self.config:
            raise KeyError(
                f"Camera not configured: {camera_id}"
            )

        return self.config[camera_id]