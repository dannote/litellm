import json
import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.yandex.chat.transformation import YandexChatConfig


class TestYandexTransform:
    def setup_method(self):
        self.config = YandexChatConfig()
        self.model = "yandex/yandexgpt/latest"
        self.logging_obj = MagicMock()

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI parameters are returned correctly."""
        params = self.config.get_supported_openai_params(self.model)

        assert "stream" in params
        assert "temperature" in params
        assert "max_tokens" in params
        assert "max_completion_tokens" in params
        assert "tools" in params
        assert "tool_choice" in params
        assert "extra_headers" in params

    def test_map_openai_params_basic(self):
        """Test that basic parameters are correctly mapped."""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "stream": True,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 200
        assert result["stream"] is True

    def test_map_openai_params_max_completion_tokens(self):
        """Test that max_completion_tokens is mapped to max_tokens."""
        test_params = {
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result["max_tokens"] == 256

    def test_convert_messages_basic(self):
        """Test basic message conversion from OpenAI to Yandex format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = self.config._convert_messages_to_yandex_format(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[0]["text"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"
        assert result[1]["text"] == "Hello!"
        assert result[2]["role"] == "assistant"
        assert result[2]["text"] == "Hi there!"

    def test_convert_messages_multipart_content(self):
        """Test message conversion with multi-part content."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ],
            }
        ]

        result = self.config._convert_messages_to_yandex_format(messages)

        assert len(result) == 1
        assert result[0]["text"] == "Hello World"

    def test_convert_messages_tool_calls(self):
        """Test message conversion with assistant tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Boston"}',
                        },
                    }
                ],
            }
        ]

        result = self.config._convert_messages_to_yandex_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "toolCallList" in result[0]
        tool_calls = result[0]["toolCallList"]["toolCalls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["functionCall"]["name"] == "get_weather"
        assert tool_calls[0]["functionCall"]["arguments"] == {"location": "Boston"}

    def test_convert_messages_tool_result(self):
        """Test message conversion with tool result."""
        messages = [
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temperature": "72F"}',
            }
        ]

        result = self.config._convert_messages_to_yandex_format(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "toolResultList" in result[0]
        tool_results = result[0]["toolResultList"]["toolResults"]
        assert len(tool_results) == 1
        assert tool_results[0]["functionResult"]["name"] == "get_weather"

    def test_convert_tools_to_yandex_format(self):
        """Test tool conversion from OpenAI to Yandex format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        result = self.config._convert_tools_to_yandex_format(tools)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the current weather"
        assert "properties" in result[0]["function"]["parameters"]

    def test_get_model_uri_with_prefix(self):
        """Test model URI construction with yandex/ prefix."""
        litellm_params = {"folder_id": "b1g12345"}

        result = self.config._get_model_uri("yandex/yandexgpt/latest", litellm_params)

        assert result == "gpt://b1g12345/yandexgpt/latest"

    def test_get_model_uri_without_prefix(self):
        """Test model URI construction without yandex/ prefix."""
        litellm_params = {"folder_id": "b1g12345"}

        result = self.config._get_model_uri("yandexgpt-lite/latest", litellm_params)

        assert result == "gpt://b1g12345/yandexgpt-lite/latest"

    def test_get_model_uri_missing_folder_id(self):
        """Test that missing folder_id raises an error."""
        litellm_params = {}

        with pytest.raises(ValueError, match="folder_id is required"):
            self.config._get_model_uri("yandexgpt/latest", litellm_params)

    def test_transform_request_basic(self):
        """Test basic request transformation."""
        import litellm

        # Mock the config
        litellm.YandexChatConfig = MagicMock()
        litellm.YandexChatConfig.get_config.return_value = {}

        messages = [{"role": "user", "content": "Hello!"}]
        optional_params = {"temperature": 0.7, "max_tokens": 100}
        litellm_params = {"folder_id": "b1g12345"}
        headers = {}

        result = self.config.transform_request(
            model="yandexgpt/latest",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert result["modelUri"] == "gpt://b1g12345/yandexgpt/latest"
        assert result["completionOptions"]["temperature"] == 0.7
        assert result["completionOptions"]["maxTokens"] == "100"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["text"] == "Hello!"

    def test_transform_request_with_tools(self):
        """Test request transformation with tools."""
        import litellm

        # Mock the config
        litellm.YandexChatConfig = MagicMock()
        litellm.YandexChatConfig.get_config.return_value = {}

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

        result = self.config.transform_request(
            model="yandexgpt/latest",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert "tools" in result
        assert result["toolChoice"]["mode"] == "AUTO"

    def test_transform_response_basic(self):
        """Test basic response transformation."""
        import litellm
        from litellm.types.utils import ModelResponse

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
                "modelVersion": "1.0",
            }
        }
        mock_response.status_code = 200

        model_response = ModelResponse()

        result = self.config.transform_response(
            model="yandex/yandexgpt/latest",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
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

    def test_transform_response_with_tool_calls(self):
        """Test response transformation with tool calls."""
        import litellm
        from litellm.types.utils import ModelResponse

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

        result = self.config.transform_response(
            model="yandex/yandexgpt/latest",
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=self.logging_obj,
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

    def test_validate_environment(self):
        """Test environment validation sets correct headers."""
        headers = {}
        result = self.config.validate_environment(
            headers=headers,
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert result["Authorization"] == "Api-Key test-api-key"
        assert result["Content-Type"] == "application/json"

    def test_get_error_class(self):
        """Test that error class is correctly returned."""
        from litellm.llms.yandex.common_utils import YandexError

        error = self.config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={},
        )

        assert isinstance(error, YandexError)
        assert error.status_code == 400
        assert error.message == "Test error"
