from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dspy import Prediction
from dspy.streaming import StreamResponse

from libraries.communication.streaming import (
    get_conversation_string,
    stream_dspy_response,
    validate_conversation,
)

# ---------------------------------------------------------------------------
# get_conversation_string
# ---------------------------------------------------------------------------


class TestGetConversationString:

    def test_empty_list_returns_empty_string(self):
        assert get_conversation_string([]) == ""

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = get_conversation_string(messages)
        assert result == "user: Hello\n"

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = get_conversation_string(messages)
        assert "user: Hello" in result
        assert "assistant: Hi there" in result
        assert result.endswith("\n")

    def test_missing_content_skips_message(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"},  # missing content
            {"role": "user", "content": "Still here?"},
        ]
        result = get_conversation_string(messages)
        assert "user: Hello" in result
        assert "user: Still here?" in result
        assert "assistant" not in result

    def test_missing_role_defaults_to_user(self):
        messages = [{"content": "Hello"}]
        result = get_conversation_string(messages)
        assert result == "User: Hello\n"

    def test_special_characters_preserved(self):
        messages = [{"role": "user", "content": "Cost is $2,500 & tax!"}]
        result = get_conversation_string(messages)
        assert "$2,500 & tax!" in result

    def test_multiline_content_preserved(self):
        messages = [{"role": "user", "content": "Line1\nLine2\nLine3"}]
        result = get_conversation_string(messages)
        assert "Line1\nLine2\nLine3" in result

    def test_with_tracer_creates_span(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        messages = [{"role": "user", "content": "Hello"}]
        result = get_conversation_string(messages, tracer=mock_tracer)

        mock_tracer.start_as_current_span.assert_called_once_with("get_conversation_string")
        assert "user: Hello" in result

    def test_without_tracer_still_works(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = get_conversation_string(messages, tracer=None)
        assert "user: Hello" in result


# ---------------------------------------------------------------------------
# validate_conversation
# ---------------------------------------------------------------------------


class TestValidateConversation:

    def test_valid_conversation_passes(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        validate_conversation("User: Hello, I need help with my policy", mock_tracer, mock_span)

    def test_empty_string_raises(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        with pytest.raises(ValueError, match="too short"):
            validate_conversation("", mock_tracer, mock_span)

    def test_whitespace_only_raises(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        with pytest.raises(ValueError, match="too short"):
            validate_conversation("    ", mock_tracer, mock_span)

    def test_very_short_string_raises(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        with pytest.raises(ValueError, match="too short"):
            validate_conversation("Hi", mock_tracer, mock_span)

    def test_exactly_5_chars_passes(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        validate_conversation("Hello", mock_tracer, mock_span)

    def test_sets_span_attributes(self):
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        validate_conversation("Hello world", mock_tracer, mock_span)
        mock_span.set_attribute.assert_any_call("conversation_length", 11)


# ---------------------------------------------------------------------------
# stream_dspy_response
# ---------------------------------------------------------------------------


class TestStreamDspyResponse:

    @pytest.mark.asyncio
    async def test_stream_start_published(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def empty_gen():
            return
            yield  # make it an async generator

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            await stream_dspy_response(
                chunks=empty_gen(),
                agent_name="TestAgent",
                connection_id="conn-1",
                message_id="msg-1",
                stream_channel=mock_channel,
                tracer=mock_tracer,
            )

        call_types = [call[0][0].type for call in mock_channel.publish.call_args_list]
        assert "agent_message_stream_start" in call_types

    @pytest.mark.asyncio
    async def test_stream_chunks_published(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def chunk_gen():
            yield StreamResponse(
                predict_name="test",
                signature_field_name="response",
                chunk="Hello ",
                is_last_chunk=False,
            )
            yield StreamResponse(
                predict_name="test",
                signature_field_name="response",
                chunk="world",
                is_last_chunk=True,
            )
            yield Prediction(final_response="Hello world")

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            await stream_dspy_response(
                chunks=chunk_gen(),
                agent_name="TestAgent",
                connection_id="conn-1",
                message_id="msg-1",
                stream_channel=mock_channel,
                tracer=mock_tracer,
            )

        call_types = [call[0][0].type for call in mock_channel.publish.call_args_list]
        assert call_types.count("agent_message_stream_chunk") == 2
        assert "agent_message_stream_start" in call_types
        assert "agent_message_stream_end" in call_types

    @pytest.mark.asyncio
    async def test_stream_end_contains_final_response(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def gen():
            yield Prediction(final_response="Your premium is $120")

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            await stream_dspy_response(
                chunks=gen(),
                agent_name="Billing",
                connection_id="conn-1",
                message_id="msg-1",
                stream_channel=mock_channel,
                tracer=mock_tracer,
            )

        end_calls = [
            c
            for c in mock_channel.publish.call_args_list
            if c[0][0].type == "agent_message_stream_end"
        ]
        assert len(end_calls) == 1
        assert end_calls[0][0][0].data["message"] == "Your premium is $120"
        assert end_calls[0][0][0].data["agent"] == "Billing"

    @pytest.mark.asyncio
    async def test_completed_marker_stripped(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def gen():
            yield Prediction(final_response="Done [[ ## completed ## ]]")

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            await stream_dspy_response(
                chunks=gen(),
                agent_name="Test",
                connection_id="conn-1",
                message_id="msg-1",
                stream_channel=mock_channel,
                tracer=mock_tracer,
            )

        end_calls = [
            c
            for c in mock_channel.publish.call_args_list
            if c[0][0].type == "agent_message_stream_end"
        ]
        assert end_calls[0][0][0].data["message"] == "Done"

    @pytest.mark.asyncio
    async def test_error_in_generator_publishes_error_message(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def error_gen():
            raise RuntimeError("DSPy failure")
            yield  # make it an async generator

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            await stream_dspy_response(
                chunks=error_gen(),
                agent_name="Claims",
                connection_id="conn-1",
                message_id="msg-1",
                stream_channel=mock_channel,
                tracer=mock_tracer,
            )

        end_calls = [
            c
            for c in mock_channel.publish.call_args_list
            if c[0][0].type == "agent_message_stream_end"
        ]
        assert len(end_calls) == 1
        assert "error" in end_calls[0][0][0].data["message"].lower()

    @pytest.mark.asyncio
    async def test_custom_span_name(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        async def gen():
            return
            yield

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            await stream_dspy_response(
                chunks=gen(),
                agent_name="Test",
                connection_id="conn-1",
                message_id="msg-1",
                stream_channel=mock_channel,
                tracer=mock_tracer,
                span_name="custom_stream",
            )

        mock_tracer.start_as_current_span.assert_called_once_with("custom_stream")

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        mock_channel = AsyncMock()
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        import asyncio

        async def cancel_gen():
            raise asyncio.CancelledError()
            yield

        with patch(
            "libraries.communication.streaming.format_span_as_traceparent",
            return_value=("tp", "ts"),
        ):
            with pytest.raises(asyncio.CancelledError):
                await stream_dspy_response(
                    chunks=cancel_gen(),
                    agent_name="Test",
                    connection_id="conn-1",
                    message_id="msg-1",
                    stream_channel=mock_channel,
                    tracer=mock_tracer,
                )
