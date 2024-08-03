import sys
import tempfile
from io import StringIO
from unittest.mock import MagicMock

from modelz.client import ModelzClient, ModelzResponse
from modelz.cmd import console, inference
from modelz.serde import RawSerde


def test_inference(monkeypatch):
    # Mocking the behavior of `ModelzClient.inference`
    params_dict = {"key": "value"}
    params_as_saved_in_file = b"Mocked Response" + b" saved in file"
    # output_file = "output.bin"
    # use a temp file instead
    output_file = tempfile.NamedTemporaryFile().name
    output_written_message = f"result has been written in {output_file}\n"

    try:

        def mock_inference(self, params, serde):
            mock_response = ModelzResponse(
                resp=MagicMock(status_code=200, content=params),
                serde=RawSerde(),
            )
            return mock_response

        # Patching the `client.inference` method with the mock
        monkeypatch.setattr(ModelzClient, "inference", mock_inference)
        mock_init = MagicMock()
        mock_init.return_value = None
        monkeypatch.setattr(ModelzClient, "__init__", mock_init)

        # Creating a StringIO object to capture the console output
        console_output = StringIO()
        console.file = console_output

        # Running the inference function with the mocked client and write to file
        inference(
            deployment="deployment_id",
            params=params_dict,
            read_stdin=False,
        )

        # Asserting the console output
        expected_output = "{'key': 'value'}\n"
        assert console_output.getvalue() == expected_output

        # sys.stdin = StringIO(stdin_input)
        # Mocking the behavior of `sys.stdin.buffer.read()`
        mock_stdin_read = MagicMock(return_value=params_as_saved_in_file)

        # Patching the `sys.stdin.buffer.read()` with the mock
        monkeypatch.setattr(sys.stdin.buffer, "read", mock_stdin_read)

        console_output = StringIO()
        console.file = console_output

        inference(
            deployment="deployment_id",
            read_stdin=True,
            write_file=output_file,
        )

        # Asserting the file content
        with open(output_file, "rb") as file:
            file_content = file.read()
            assert file_content == params_as_saved_in_file

        # Asserting the console output
        assert console_output.getvalue() == output_written_message

    except Exception as e:
        raise e

    finally:
        # Resetting the sys.stdin
        # sys.stdin = sys.__stdin__
        monkeypatch.undo()

        # remove output_file if exists
        import os

        if os.path.exists(output_file):
            os.remove(output_file)
