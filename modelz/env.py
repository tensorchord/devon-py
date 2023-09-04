from __future__ import annotations

import os

PREFIX: str = "MODELZ_"


class EnvConfig:
    def __init__(self) -> None:
        self.api_key: str
        self.host: str = "https://{}.modelz.io/"
        self.disable_rich: bool = True
        self.update_from_env()

    def update_from_env(self):
        for key in ("api_key", "host", "disable_rich"):
            val = os.environ.get(f"{PREFIX}{key.upper()}")
            if val is not None:
                setattr(self, key, type(key)(val))
                print(f"update {key} from env: {val}")
