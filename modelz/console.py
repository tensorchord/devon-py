import json
import re

import httpx
from rich.console import Console

from modelz.env import EnvConfig

config = EnvConfig()


class ModelzConsole:
    class Status:
        def __init__(self, msg) -> None:
            self.msg = msg

        def __enter__(self):
            print(self.msg)

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def __init__(self) -> None:
        self._remove_style = re.compile(r"\[.*?\]")
        self.formatter = lambda msg: self._remove_style.sub("", msg)

    def status(self, msg: str):
        return self.Status(self.formatter(msg))

    def print(self, msg: str):
        print(self.formatter(msg))


console = ModelzConsole() if config.disable_rich else Console()


def jsonFormattedPrint(resp: httpx.Response):
    try:
        formatted = json.dumps(json.loads(resp.content.decode()), indent=2)
    except Exception:
        formatted = resp.content.decode()
    finally:
        console.print(formatted)
