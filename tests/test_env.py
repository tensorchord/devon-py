import os

import pytest

from modelz.env import EnvConfig


@pytest.fixture
def config():
    os.environ["MODELZ_API_KEY"] = "abc123"
    os.environ["MODELZ_HOST"] = "https://custom.host/"
    yield EnvConfig()
    os.environ.pop("MODELZ_API_KEY")
    os.environ.pop("MODELZ_HOST")


def test_update_from_env(config):
    assert config.api_key == "abc123"
    assert config.host == "https://custom.host/"


def test_default():
    with pytest.raises(AttributeError):
        config = EnvConfig()
        assert config.api_key is None
        assert config.host == "https://{}.modelz.io/"
