import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import ModelResponseIterator as YandexModelResponseIterator
from ..common_utils import YandexError
from ..common_utils import YandexModelInfo
from ..common_utils import validate_environment as yandex_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class YandexChatConfig(BaseConfig):
    """
    Configuration class for Yandex Cloud Text Generation API.

    Yandex API Reference: https://yandex.cloud/ru/docs/ai-studio/text-generation/api-ref/TextGeneration/completion

    Args:
        temperature (float, optional): Controls randomness in generation. Range 0-1, default 0.3.
        max_tokens (int, optional): Maximum number of tokens to generate.
        stream (bool, optional): Enable streaming responses.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return yandex_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
        )

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            elif param == "temperature":
                optional_params["temperature"] = value
            elif param == "max_tokens":
                optional_params["max_tokens"] = value
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param == "tools":
                optional_params["tools"] = value
            elif param == "tool_choice":
                optional_params["tool_choice"] = value
        return optional_params

    def _convert_messages_to_yandex_format(
        self, messages: List[AllMessageValues]
    ) -> List[dict]:
        """Convert OpenAI message format to Yandex format."""
        yandex_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle tool call results
            if role == "tool":
                # Yandex expects tool results in a specific format
                yandex_messages.append(
                    {
                        "role": "assistant",
                        "toolResultList": {
                            "toolResults": [
                                {
                                    "functionResult": {
                                        "name": msg.get("name", ""),
                                        "content": content
                                        if isinstance(content, str)
                                        else json.dumps(content),
                                    }
                                }
                            ]
                        },
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                # Convert assistant message with tool calls
                tool_calls = msg.get("tool_calls", [])
                yandex_tool_calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    yandex_tool_calls.append(
                        {"functionCall": {"name": func.get("name", ""), "arguments": args}}
                    )
                yandex_messages.append(
                    {
                        "role": "assistant",
                        "toolCallList": {"toolCalls": yandex_tool_calls},
                    }
                )
            else:
                # Standard message
                if isinstance(content, list):
                    # Handle multi-part content (extract text parts)
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = " ".join(text_parts)

                yandex_messages.append({"role": role, "text": content})

        return yandex_messages

    def _convert_tools_to_yandex_format(self, tools: List[dict]) -> List[dict]:
        """Convert OpenAI tools format to Yandex format."""
        yandex_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                yandex_tools.append(
                    {
                        "function": {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    }
                )
        return yandex_tools

    def _get_model_uri(self, model: str, litellm_params: dict) -> str:
        """
        Construct Yandex modelUri from model name and folder_id.

        Model format: yandex/model-name -> gpt://{folder_id}/model-name
        """
        # Get folder_id from litellm_params or environment
        folder_id = litellm_params.get("folder_id") or YandexModelInfo.get_folder_id()

        if not folder_id:
            raise ValueError(
                "Yandex folder_id is required. Set YANDEX_FOLDER_ID environment variable or pass folder_id parameter."
            )

        # Strip 'yandex/' prefix if present
        model_name = model
        if model_name.startswith("yandex/"):
            model_name = model_name[7:]

        return f"gpt://{folder_id}/{model_name}"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Transform OpenAI-style request to Yandex format."""
        # Load config defaults
        for k, v in litellm.YandexChatConfig.get_config().items():
            if k not in optional_params:
                optional_params[k] = v

        # Build Yandex request
        model_uri = self._get_model_uri(model, litellm_params)

        # Build completion options
        completion_options = {}
        if optional_params.get("stream") is not None:
            completion_options["stream"] = optional_params["stream"]
        if optional_params.get("temperature") is not None:
            completion_options["temperature"] = optional_params["temperature"]
        if optional_params.get("max_tokens") is not None:
            completion_options["maxTokens"] = str(optional_params["max_tokens"])

        request_data = {
            "modelUri": model_uri,
            "completionOptions": completion_options,
            "messages": self._convert_messages_to_yandex_format(messages),
        }

        # Add tools if present
        if optional_params.get("tools"):
            request_data["tools"] = self._convert_tools_to_yandex_format(
                optional_params["tools"]
            )

        # Add tool choice if present
        if optional_params.get("tool_choice"):
            tool_choice = optional_params["tool_choice"]
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    request_data["toolChoice"] = {"mode": "AUTO"}
                elif tool_choice == "none":
                    request_data["toolChoice"] = {"mode": "NONE"}
                elif tool_choice == "required":
                    request_data["toolChoice"] = {"mode": "REQUIRED"}
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function":
                    request_data["toolChoice"] = {
                        "mode": "TOOL",
                        "toolName": tool_choice.get("function", {}).get("name", ""),
                    }

        return request_data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """Transform Yandex response to OpenAI format."""
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise YandexError(
                message=raw_response.text, status_code=raw_response.status_code
            )

        # Extract response from Yandex format
        result = raw_response_json.get("result", raw_response_json)
        alternatives = result.get("alternatives", [])

        if not alternatives:
            raise YandexError(
                message="No alternatives in Yandex response",
                status_code=raw_response.status_code,
            )

        alt = alternatives[0]
        message = alt.get("message", {})

        # Set content
        content = message.get("text", "")
        model_response.choices[0].message.content = content  # type: ignore

        # Handle tool calls
        tool_call_list = message.get("toolCallList", {})
        tool_calls_data = tool_call_list.get("toolCalls", [])
        if tool_calls_data:
            tool_calls = []
            for i, tc in enumerate(tool_calls_data):
                func_call = tc.get("functionCall", {})
                tool_call = {
                    "id": f"call_{i}_{hash(func_call.get('name', ''))}",
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("arguments", {})),
                    },
                }
                tool_calls.append(tool_call)

            _message = litellm.Message(
                tool_calls=tool_calls,
                content=content if content else None,
            )
            model_response.choices[0].message = _message  # type: ignore

        # Set finish reason
        status = alt.get("status", "ALTERNATIVE_STATUS_COMPLETE")
        if status == "ALTERNATIVE_STATUS_TOOL_CALLS":
            model_response.choices[0].finish_reason = "tool_calls"
        elif "TRUNCATED" in status:
            model_response.choices[0].finish_reason = "length"
        else:
            model_response.choices[0].finish_reason = "stop"

        # Parse usage
        usage_data = result.get("usage", {})
        prompt_tokens = int(usage_data.get("inputTextTokens", 0))
        completion_tokens = int(usage_data.get("completionTokens", 0))

        model_response.created = int(time.time())
        model_response.model = model

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)

        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return YandexModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return YandexError(status_code=status_code, message=error_message)
