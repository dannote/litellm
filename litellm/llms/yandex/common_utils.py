import json
import os
from typing import List, Optional

import httpx

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    ProviderSpecificModelInfo,
)


class YandexError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[httpx.Headers] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST",
            url="https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


class YandexModelInfo(BaseLLMModelInfo):
    def get_provider_info(
        self,
        model: str,
    ) -> Optional[ProviderSpecificModelInfo]:
        return None

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        return []

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or os.environ.get("YANDEX_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        )

    @staticmethod
    def get_folder_id(folder_id: Optional[str] = None) -> Optional[str]:
        return folder_id or os.environ.get("YANDEX_FOLDER_ID")

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
        return {}

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return None


def validate_environment(
    headers: dict,
    model: str,
    messages: List[AllMessageValues],
    optional_params: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Return headers to use for Yandex completion request.

    Yandex API expects:
    {
        "Authorization": "Api-Key {api_key}",
        "Content-Type": "application/json"
    }
    """
    headers.update(
        {
            "Content-Type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    return headers


class ModelResponseIterator:
    """Iterator for Yandex streaming responses."""

    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        """Parse Yandex streaming chunk to GenericStreamingChunk."""
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            # Yandex streaming format has 'result' containing alternatives
            result = chunk.get("result", {})
            alternatives = result.get("alternatives", [])

            if alternatives:
                alt = alternatives[0]
                message = alt.get("message", {})
                text = message.get("text", "")

                # Check for tool calls
                tool_calls = message.get("toolCallList", {}).get("toolCalls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    func_call = tc.get("functionCall", {})
                    tool_use = ChatCompletionToolCallChunk(
                        id=f"call_{hash(func_call.get('name', ''))}",
                        type="function",
                        function={
                            "name": func_call.get("name", ""),
                            "arguments": json.dumps(func_call.get("arguments", {})),
                        },
                        index=0,
                    )

                status = alt.get("status", "")
                if status in (
                    "ALTERNATIVE_STATUS_COMPLETE",
                    "ALTERNATIVE_STATUS_FINAL",
                    "ALTERNATIVE_STATUS_TRUNCATED_FINAL",
                ):
                    is_finished = True
                    finish_reason = "stop" if "TRUNCATED" not in status else "length"
                elif status == "ALTERNATIVE_STATUS_TOOL_CALLS":
                    is_finished = True
                    finish_reason = "tool_calls"

            # Parse usage if present
            usage_data = result.get("usage", {})
            if usage_data:
                input_tokens = int(usage_data.get("inputTextTokens", 0))
                output_tokens = int(usage_data.get("completionTokens", 0))
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=0,
                provider_specific_fields=provider_specific_fields,
            )

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    def convert_str_chunk_to_generic_chunk(self, chunk: str) -> GenericStreamingChunk:
        """Convert a string chunk to a GenericStreamingChunk."""
        str_line = chunk
        if isinstance(chunk, bytes):
            str_line = chunk.decode("utf-8")

        data_json = json.loads(str_line)
        return self.chunk_parser(chunk=data_json)

    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
