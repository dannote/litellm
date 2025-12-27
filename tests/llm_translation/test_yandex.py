import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.yandex.chat.transformation import YandexChatConfig
from litellm.types.utils import ModelResponse


class TestYandexTransformationIntegration:
    """Test Yandex provider transformation logic."""

    def test_request_transformation(self):
        """Test that request is correctly transformed to Yandex format."""
        config = YandexChatConfig()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        optional_params = {"temperature": 0.7, "max_tokens": 100}
        litellm_params = {"folder_id": "b1g12345"}
        headers = {}

        # Mock the config getter
        with patch.object(litellm.YandexChatConfig, "get_config", return_value={}):
            result = config.transform_request(
                model="yandexgpt/latest",
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )

        # Verify Yandex request format
        assert result["modelUri"] == "gpt://b1g12345/yandexgpt/latest"
        assert result["completionOptions"]["temperature"] == 0.7
        assert result["completionOptions"]["maxTokens"] == "100"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["text"] == "You are helpful."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["text"] == "Hello!"

    def test_request_transformation_with_tools(self):
        """Test that tools are correctly transformed to Yandex format."""
        config = YandexChatConfig()

        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        optional_params = {"tools": tools, "tool_choice": "auto"}
        litellm_params = {"folder_id": "b1g12345"}
        headers = {}

        with patch.object(litellm.YandexChatConfig, "get_config", return_value={}):
            result = config.transform_request(
                model="yandexgpt/latest",
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )

        assert "tools" in result
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert result["toolChoice"]["mode"] == "AUTO"

    def test_response_transformation(self):
        """Test that Yandex response is correctly transformed to OpenAI format."""
        config = YandexChatConfig()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {
                "alternatives": [
                    {
                        "message": {"role": "assistant", "text": "Hello!"},
                        "status": "ALTERNATIVE_STATUS_COMPLETE",
                    }
                ],
                "usage": {
                    "inputTextTokens": "10",
                    "completionTokens": "5",
                    "totalTokens": "15",
                },
            }
        }
        mock_response.status_code = 200

        model_response = ModelResponse()

        result = config.transform_response(
            model="yandex/yandexgpt/latest",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_response_transformation_with_tool_calls(self):
        """Test that tool calls are correctly transformed."""
        config = YandexChatConfig()

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "result": {
                "alternatives": [
                    {
                        "message": {
                            "role": "assistant",
                            "text": "",
                            "toolCallList": {
                                "toolCalls": [
                                    {
                                        "functionCall": {
                                            "name": "get_weather",
                                            "arguments": {"location": "Boston"},
                                        }
                                    }
                                ]
                            },
                        },
                        "status": "ALTERNATIVE_STATUS_TOOL_CALLS",
                    }
                ],
                "usage": {
                    "inputTextTokens": "10",
                    "completionTokens": "5",
                    "totalTokens": "15",
                },
            }
        }
        mock_response.status_code = 200

        model_response = ModelResponse()

        result = config.transform_response(
            model="yandex/yandexgpt/latest",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].finish_reason == "tool_calls"
        assert len(result.choices[0].message.tool_calls) == 1
        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"location": "Boston"}


class TestYandexProviderRegistration:
    """Test that Yandex provider is properly registered."""

    def test_provider_in_llm_providers(self):
        """Test YANDEX is in LlmProviders enum."""
        from litellm.types.utils import LlmProviders

        assert hasattr(LlmProviders, "YANDEX")
        assert LlmProviders.YANDEX.value == "yandex"

    def test_provider_config_manager_returns_yandex_config(self):
        """Test ProviderConfigManager returns YandexChatConfig for Yandex."""
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            "yandex/test", LlmProviders.YANDEX
        )
        assert isinstance(config, YandexChatConfig)


class TestYandexModelInfo:
    """Test Yandex model info utilities."""

    def test_get_api_key_from_param(self):
        """Test API key retrieval from parameter."""
        from litellm.llms.yandex.common_utils import YandexModelInfo

        result = YandexModelInfo.get_api_key("test-key")
        assert result == "test-key"

    def test_get_api_key_from_env(self):
        """Test API key retrieval from environment."""
        from litellm.llms.yandex.common_utils import YandexModelInfo

        os.environ["YANDEX_API_KEY"] = "env-test-key"
        try:
            result = YandexModelInfo.get_api_key()
            assert result == "env-test-key"
        finally:
            del os.environ["YANDEX_API_KEY"]

    def test_get_folder_id_from_param(self):
        """Test folder ID retrieval from parameter."""
        from litellm.llms.yandex.common_utils import YandexModelInfo

        result = YandexModelInfo.get_folder_id("b1g12345")
        assert result == "b1g12345"

    def test_get_folder_id_from_env(self):
        """Test folder ID retrieval from environment."""
        from litellm.llms.yandex.common_utils import YandexModelInfo

        os.environ["YANDEX_FOLDER_ID"] = "env-folder-id"
        try:
            result = YandexModelInfo.get_folder_id()
            assert result == "env-folder-id"
        finally:
            del os.environ["YANDEX_FOLDER_ID"]

    def test_get_api_base_default(self):
        """Test default API base URL."""
        from litellm.llms.yandex.common_utils import YandexModelInfo

        result = YandexModelInfo.get_api_base()
        assert (
            result
            == "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        )

    def test_get_api_base_custom(self):
        """Test custom API base URL."""
        from litellm.llms.yandex.common_utils import YandexModelInfo

        result = YandexModelInfo.get_api_base("https://custom.api")
        assert result == "https://custom.api"


class TestYandexError:
    """Test Yandex error handling."""

    def test_yandex_error_creation(self):
        """Test YandexError instantiation."""
        from litellm.llms.yandex.common_utils import YandexError

        error = YandexError(status_code=400, message="Bad request")

        assert error.status_code == 400
        assert error.message == "Bad request"
        assert error.request is not None
        assert error.response is not None


class TestYandexValidateEnvironment:
    """Test Yandex environment validation."""

    def test_validate_sets_authorization_header(self):
        """Test that validate_environment sets correct headers."""
        config = YandexChatConfig()

        headers = {}
        result = config.validate_environment(
            headers=headers,
            model="yandexgpt/latest",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="my-api-key",
        )

        assert result["Authorization"] == "Api-Key my-api-key"
        assert result["Content-Type"] == "application/json"
