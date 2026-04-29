import importlib
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class LLMConfigTests(unittest.TestCase):
    def reload_modules(self):
        import config
        from tests import test_llm_api

        importlib.reload(config)
        return importlib.reload(test_llm_api)

    def reload_config(self):
        import config

        return importlib.reload(config)

    @mock.patch.dict(
        os.environ,
        {
            "KGQA_LOAD_DOTENV": "0",
            "LLM_API_KEY": "test-secret",
            "LLM_URL": "https://example.test/chat/completions",
            "LLM_MODEL": "test-model",
        },
        clear=True,
    )
    def test_llm_api_config_comes_from_environment(self):
        llm_api = self.reload_modules()

        config = llm_api.get_api_config()

        self.assertEqual(config.api_key, "test-secret")
        self.assertEqual(config.url, "https://example.test/chat/completions")
        self.assertEqual(config.model, "test-model")

    @mock.patch.dict(os.environ, {"KGQA_LOAD_DOTENV": "0"}, clear=True)
    def test_llm_api_check_skips_network_without_api_key(self):
        llm_api = self.reload_modules()

        with mock.patch.object(llm_api.requests, "post") as post, redirect_stdout(io.StringIO()):
            success = llm_api.check_llm_api()

        self.assertFalse(success)
        post.assert_not_called()

    @mock.patch.dict(os.environ, {"KGQA_LOAD_DOTENV": "1"}, clear=True)
    def test_llm_config_reloads_dotenv_when_file_appears(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            try:
                config = self.reload_config()
                self.assertFalse(config.has_llm_api_key(config.get_llm_config().api_key))

                with open(".env", "w", encoding="utf-8") as env_file:
                    env_file.write("LLM_API_KEY=dotenv-secret\n")
                    env_file.write("LLM_URL=https://dotenv.test/chat\n")
                    env_file.write("LLM_MODEL=dotenv-model\n")

                llm_config = config.get_llm_config()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(llm_config.api_key, "dotenv-secret")
        self.assertEqual(llm_config.url, "https://dotenv.test/chat")
        self.assertEqual(llm_config.model, "dotenv-model")

    @mock.patch.dict(
        os.environ,
        {
            "KGQA_LOAD_DOTENV": "0",
            "LLM_API_KEY": "status-secret",
            "LLM_URL": "https://status.test/chat",
            "LLM_MODEL": "status-model",
        },
        clear=True,
    )
    def test_llm_status_reports_config_without_exposing_secret(self):
        config = self.reload_config()

        status = config.get_llm_status()

        self.assertEqual(
            status,
            {
                "configured": True,
                "url": "https://status.test/chat",
                "model": "status-model",
            },
        )
        self.assertNotIn("api_key", status)


if __name__ == "__main__":
    unittest.main()
