"""
Tests for Anthropic OAuth pass-through support.

Validates that client-provided authentication headers (Authorization, x-api-key)
take precedence over server-configured API keys, enabling Claude Code OAuth tokens
to pass through the proxy.

Related: https://github.com/BerriAI/litellm/issues/13380
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    anthropic_proxy_route,
)


class TestAnthropicAuthHeaders:
    """Test suite for Anthropic authentication header handling."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock(spec=Request)
        request.url = MagicMock()
        request.url.path = "/anthropic/v1/messages"
        return request

    @pytest.fixture
    def mock_response(self):
        """Create a mock FastAPI response."""
        return MagicMock()

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_client_authorization_header_takes_precedence(
        self, mock_request, mock_response, mock_user_api_key_dict
    ):
        """
        Test that client-provided Authorization header takes precedence.
        This enables OAuth tokens (sk-ant-oat01-*) to pass through.
        """
        # Client provides Authorization header (OAuth token)
        mock_request.headers = {
            "authorization": "Bearer sk-ant-oat01-test-oauth-token",
            "content-type": "application/json",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_router, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new_callable=AsyncMock,
        ) as mock_streaming, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            # Server has an API key configured
            mock_router.get_credentials.return_value = "server-api-key"
            mock_streaming.return_value = False

            mock_endpoint_func = AsyncMock(return_value={"status": "ok"})
            mock_create_route.return_value = mock_endpoint_func

            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify custom_headers is empty (client auth takes precedence)
            call_kwargs = mock_create_route.call_args.kwargs
            assert call_kwargs["custom_headers"] == {}

    @pytest.mark.asyncio
    async def test_client_x_api_key_takes_precedence(
        self, mock_request, mock_response, mock_user_api_key_dict
    ):
        """
        Test that client-provided x-api-key header takes precedence.
        """
        # Client provides x-api-key header
        mock_request.headers = {
            "x-api-key": "client-provided-api-key",
            "content-type": "application/json",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_router, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new_callable=AsyncMock,
        ) as mock_streaming, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            # Server has an API key configured
            mock_router.get_credentials.return_value = "server-api-key"
            mock_streaming.return_value = False

            mock_endpoint_func = AsyncMock(return_value={"status": "ok"})
            mock_create_route.return_value = mock_endpoint_func

            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify custom_headers is empty (client auth takes precedence)
            call_kwargs = mock_create_route.call_args.kwargs
            assert call_kwargs["custom_headers"] == {}

    @pytest.mark.asyncio
    async def test_server_api_key_used_when_no_client_auth(
        self, mock_request, mock_response, mock_user_api_key_dict
    ):
        """
        Test that server API key is used when client provides no authentication.
        """
        # Client provides no authentication headers
        mock_request.headers = {
            "content-type": "application/json",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_router, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new_callable=AsyncMock,
        ) as mock_streaming, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            # Server has an API key configured
            mock_router.get_credentials.return_value = "server-api-key"
            mock_streaming.return_value = False

            mock_endpoint_func = AsyncMock(return_value={"status": "ok"})
            mock_create_route.return_value = mock_endpoint_func

            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify server API key is injected
            call_kwargs = mock_create_route.call_args.kwargs
            assert call_kwargs["custom_headers"] == {"x-api-key": "server-api-key"}

    @pytest.mark.asyncio
    async def test_no_headers_when_no_auth_anywhere(
        self, mock_request, mock_response, mock_user_api_key_dict
    ):
        """
        Test that no x-api-key header is added when neither client nor server has auth.
        """
        # Client provides no authentication headers
        mock_request.headers = {
            "content-type": "application/json",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_router, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new_callable=AsyncMock,
        ) as mock_streaming, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            # Server has no API key configured
            mock_router.get_credentials.return_value = None
            mock_streaming.return_value = False

            mock_endpoint_func = AsyncMock(return_value={"status": "ok"})
            mock_create_route.return_value = mock_endpoint_func

            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify custom_headers is empty
            call_kwargs = mock_create_route.call_args.kwargs
            assert call_kwargs["custom_headers"] == {}

    @pytest.mark.asyncio
    async def test_both_client_headers_present(
        self, mock_request, mock_response, mock_user_api_key_dict
    ):
        """
        Test behavior when client provides both Authorization and x-api-key headers.
        Server key should not be injected.
        """
        # Client provides both headers
        mock_request.headers = {
            "authorization": "Bearer sk-ant-oat01-test-oauth-token",
            "x-api-key": "client-api-key",
            "content-type": "application/json",
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router"
        ) as mock_router, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.is_streaming_request_fn",
            new_callable=AsyncMock,
        ) as mock_streaming, patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.create_pass_through_route"
        ) as mock_create_route:
            # Server has an API key configured
            mock_router.get_credentials.return_value = "server-api-key"
            mock_streaming.return_value = False

            mock_endpoint_func = AsyncMock(return_value={"status": "ok"})
            mock_create_route.return_value = mock_endpoint_func

            await anthropic_proxy_route(
                endpoint="v1/messages",
                request=mock_request,
                fastapi_response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify custom_headers is empty (client auth takes precedence)
            call_kwargs = mock_create_route.call_args.kwargs
            assert call_kwargs["custom_headers"] == {}
