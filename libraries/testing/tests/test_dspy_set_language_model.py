"""Tests for DSPy language model configuration."""

from unittest.mock import Mock, patch

import dspy
import pytest

from libraries.ml.dspy import (
    LenientChatAdapter,
    TrackingLM,
    dspy_set_language_model,
)


class TestTrackingLM:
    """Test the TrackingLM class."""

    @patch("dspy.LM.__init__")
    def test_tracking_lm_initialization(self, mock_super_init):
        """Test TrackingLM initialization."""
        mock_super_init.return_value = None

        # Test regular model
        lm = TrackingLM("openai/gpt-4")
        assert lm.model_name == "openai/gpt-4"
        assert lm.is_lm_studio is False
        assert lm.max_context_window == 16384
        assert lm.completion_tokens == 0
        assert lm.prompt_tokens == 0
        assert lm.total_tokens == 0

    @patch("dspy.LM.__init__")
    def test_tracking_lm_lm_studio(self, mock_super_init):
        """Test TrackingLM with LM Studio model."""
        mock_super_init.return_value = None

        # Test LM Studio model
        lm = TrackingLM("lm_studio/model", response_format="json")
        assert lm.is_lm_studio is True
        assert lm.max_context_window == 128000
        # response_format should be removed for LM Studio
        _, kwargs = mock_super_init.call_args
        assert "response_format" not in kwargs

    def test_truncate_prompt(self):
        """Test prompt truncation."""
        with patch("dspy.LM.__init__", return_value=None):
            lm = TrackingLM("test-model")
            lm.max_context_window = 100

            # Test no truncation needed
            short_prompt = "Short prompt"
            assert lm._truncate_prompt(short_prompt) == short_prompt

            # Test truncation needed (assuming 4 chars per token)
            # 100 tokens * 0.8 = 80 tokens available * 4 chars = 320 chars
            long_prompt = "x" * 400
            truncated = lm._truncate_prompt(long_prompt)
            assert truncated.startswith("...")
            assert len(truncated) < len(long_prompt)

            # Test None prompt
            assert lm._truncate_prompt(None) is None

    def test_truncate_messages(self):
        """Test messages truncation."""
        with patch("dspy.LM.__init__", return_value=None):
            lm = TrackingLM("test-model")
            lm.max_context_window = 100

            # Test short messages
            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            assert lm._truncate_messages(messages) == messages

            # Test long messages that need truncation
            long_messages = [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "x" * 200},
                {"role": "assistant", "content": "x" * 200},
                {"role": "user", "content": "x" * 200},
                {"role": "assistant", "content": "x" * 200},
                {"role": "user", "content": "Latest message"},
            ]
            truncated = lm._truncate_messages(long_messages)
            # Should keep system message and recent messages
            assert truncated[0]["role"] == "system"
            assert len(truncated) <= len(long_messages)

    @patch("dspy.settings")
    @patch("dspy.configure")
    @patch("libraries.ml.dspy.language_model.load_dotenv")
    def test_dspy_set_language_model_basic(
        self, mock_load_dotenv, mock_configure, mock_dspy_settings
    ):
        """Test basic language model setup."""
        # Create mock settings
        mock_settings = Mock()
        mock_settings.language_model = "openai/gpt-4"
        mock_settings.cache_enabled = True
        mock_settings.language_model_api_base = None

        # Mock TrackingLM by patching the class
        with patch("libraries.ml.dspy.language_model.TrackingLM") as mock_tracking_lm:
            mock_lm_instance = Mock()
            mock_tracking_lm.return_value = mock_lm_instance

            # Call function
            result = dspy_set_language_model(mock_settings)

            # Verify
            mock_load_dotenv.assert_called_once()
            mock_tracking_lm.assert_called_once_with("openai/gpt-4", cache=True, api_base=None)
            mock_configure.assert_called_once()
            assert mock_configure.call_args.kwargs["lm"] is mock_lm_instance
            adapter = mock_configure.call_args.kwargs["adapter"]
            assert isinstance(adapter, LenientChatAdapter)
            assert adapter.use_json_adapter_fallback is False
            mock_dspy_settings.configure.assert_called_once_with(track_usage=True)
            assert result is mock_lm_instance

    @patch("dspy.settings")
    @patch("dspy.configure")
    @patch("libraries.ml.dspy.language_model.TrackingLM")
    def test_dspy_set_language_model_with_cache_control(
        self, mock_tracking_lm, mock_configure, mock_dspy_settings
    ):
        """Test language model setup with cache control."""
        mock_settings = Mock()
        mock_settings.language_model = "anthropic/claude-3"
        mock_settings.cache_enabled = True
        mock_settings.language_model_api_base = None

        mock_lm_instance = Mock()
        mock_tracking_lm.return_value = mock_lm_instance

        # Test with cache enabled override
        dspy_set_language_model(mock_settings, overwrite_cache_enabled=True)

        call_args = mock_tracking_lm.call_args
        assert call_args[1]["cache"] is True

        # Test with cache disabled override
        dspy_set_language_model(mock_settings, overwrite_cache_enabled=False)

        call_args = mock_tracking_lm.call_args
        assert call_args[1]["cache"] is False

    @patch("dspy.settings")
    @patch("dspy.configure")
    @patch("libraries.ml.dspy.language_model.TrackingLM")
    def test_dspy_set_language_model_with_api_base(
        self, mock_tracking_lm, mock_configure, mock_dspy_settings
    ):
        """Test language model setup with custom API base."""
        mock_settings = Mock()
        mock_settings.language_model = "lm_studio/model"
        mock_settings.cache_enabled = False
        mock_settings.language_model_api_base = "http://localhost:1234/v1"

        mock_lm_instance = Mock()
        mock_tracking_lm.return_value = mock_lm_instance

        dspy_set_language_model(mock_settings)

        mock_tracking_lm.assert_called_once_with(
            "lm_studio/model", cache=False, api_base="http://localhost:1234/v1"
        )

    @patch("dspy.settings")
    @patch("dspy.configure")
    @patch("libraries.ml.dspy.language_model.TrackingLM")
    def test_dspy_set_language_model_with_max_context(
        self, mock_tracking_lm, mock_configure, mock_dspy_settings
    ):
        """Test language model setup with max context window."""
        mock_settings = Mock()
        mock_settings.language_model = "openai/gpt-4"
        mock_settings.cache_enabled = True
        mock_settings.language_model_api_base = None
        mock_settings.max_context_window = 32000

        mock_lm_instance = Mock()
        mock_tracking_lm.return_value = mock_lm_instance

        dspy_set_language_model(mock_settings)

        # Verify max_context_window was set
        assert mock_lm_instance.max_context_window == 32000

    @patch("libraries.ml.dspy.language_model.TrackingLM")
    def test_dspy_set_language_model_error_handling(self, mock_tracking_lm):
        """Test error handling in language model setup."""
        mock_settings = Mock()
        mock_settings.language_model = "invalid/model"
        mock_settings.cache_enabled = True
        mock_settings.language_model_api_base = None

        # Simulate error when creating TrackingLM
        mock_tracking_lm.side_effect = Exception("Invalid model")

        with pytest.raises(Exception) as exc_info:
            dspy_set_language_model(mock_settings)

        assert "Invalid model" in str(exc_info.value)

    @patch("dspy.settings")
    @patch("dspy.configure")
    @patch("libraries.observability.logger.get_console_logger")
    @patch("libraries.ml.dspy.language_model.TrackingLM")
    def test_dspy_set_language_model_logging(
        self, mock_tracking_lm, mock_get_logger, mock_configure, mock_dspy_settings
    ):
        """Test logging in language model setup."""
        mock_settings = Mock()
        mock_settings.language_model = "openai/gpt-4"
        mock_settings.cache_enabled = True
        mock_settings.language_model_api_base = None

        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        mock_lm_instance = Mock()
        mock_lm_instance.max_context_window = 16384
        mock_lm_instance.is_lm_studio = False
        mock_tracking_lm.return_value = mock_lm_instance

        dspy_set_language_model(mock_settings)

        # Verify logger was created
        mock_get_logger.assert_called_once_with("dspy_language_model")

        # Verify logging was called
        assert mock_logger.info.called
        # At least some info should be logged
        assert mock_logger.info.call_count >= 1


class TestDspySetLanguageModelExtras:
    """Additional tests for the dspy_set_language_model module."""

    @patch("dspy.LM.forward")
    def test_tracking_lm_forward(self, mock_forward):
        """Test TrackingLM forward method tracks usage."""
        with patch("dspy.LM.__init__", return_value=None):
            lm = TrackingLM("test-model")

            # Mock forward result
            mock_result = Mock()
            mock_result.usage = {"completion_tokens": 50, "prompt_tokens": 100, "total_tokens": 150}
            mock_forward.return_value = mock_result

            # Call forward
            result = lm.forward("test prompt")

            # Verify usage tracking
            assert lm.completion_tokens == 50
            assert lm.prompt_tokens == 100
            assert lm.total_tokens == 150
            assert result == mock_result

    @patch("dspy.LM.__call__")
    @patch("libraries.ml.dspy.language_model.perf_counter")
    def test_tracking_lm_call_latency(self, mock_perf_counter, mock_call):
        """Test TrackingLM tracks latency."""
        with patch("dspy.LM.__init__", return_value=None):
            lm = TrackingLM("test-model")

            # Mock timing
            mock_perf_counter.side_effect = [1.0, 1.5]  # 500ms difference
            mock_call.return_value = "result"

            # Call the LM
            result = lm("test prompt")

            # Verify latency tracking
            assert lm.latency_ms == 500.0
            assert result == "result"


class TestLenientChatAdapter:
    """Test the lenient ChatAdapter used for local models."""

    def test_parses_headers_glued_on_one_line(self):
        from enum import StrEnum

        class TargetAgent(StrEnum):
            PolicyAgent = "PolicyAgent"
            ClaimsAgent = "ClaimsAgent"

        signature = dspy.Signature("chat_history -> target_agent: TargetAgent")
        signature = signature.with_updated_fields("target_agent", annotation=TargetAgent)
        adapter = LenientChatAdapter()

        result = adapter.parse(
            signature, "[[ ## target_agent ## ]]PolicyAgent[[ ## completed ## ]]"
        )

        assert result == {"target_agent": TargetAgent.PolicyAgent}

    def test_parses_well_formed_output(self):
        signature = dspy.Signature("chat_history -> response")
        adapter = LenientChatAdapter()

        result = adapter.parse(
            signature, "[[ ## response ## ]]\nHello there!\n\n[[ ## completed ## ]]\n"
        )

        assert result == {"response": "Hello there!"}

    def test_empty_dict_field_parses_as_empty_dict(self):

        signature = dspy.Signature(
            "chat_history -> next_thought: str, next_tool_name: str, next_tool_args: dict[str, Any]"
        )
        adapter = LenientChatAdapter()

        result = adapter.parse(
            signature,
            "[[ ## next_thought ## ]]\n\nDone.\n\n[[ ## next_tool_name ## ]]\n\nfinish\n\n"
            "[[ ## next_tool_args ## ]]\n\n\n\n[[ ## completed ## ]]\n",
        )

        assert result == {"next_thought": "Done.", "next_tool_name": "finish", "next_tool_args": {}}

    def test_non_empty_dict_field_is_untouched(self):

        signature = dspy.Signature("chat_history -> next_tool_args: dict[str, Any]")
        adapter = LenientChatAdapter()

        result = adapter.parse(
            signature,
            '[[ ## next_tool_args ## ]]\n{"policy_number": "A123"}\n[[ ## completed ## ]]',
        )

        assert result == {"next_tool_args": {"policy_number": "A123"}}

    def test_think_block_is_ignored(self):
        signature = dspy.Signature("chat_history -> response")
        adapter = LenientChatAdapter()

        result = adapter.parse(
            signature,
            "<think>\nThe user says hi. I should put the answer under [[ ## response ## ]].\n</think>\n"
            "[[ ## response ## ]]\nHello!\n[[ ## completed ## ]]",
        )

        assert result == {"response": "Hello!"}

    def test_format_appends_no_tool_call_instruction_to_system_prompt(self):
        from libraries.ml.dspy.language_model import NO_TOOL_CALL_INSTRUCTION

        signature = dspy.Signature("chat_history -> response")
        messages = LenientChatAdapter().format(signature, demos=[], inputs={"chat_history": "hi"})

        assert messages[0]["role"] == "system"
        assert messages[0]["content"].endswith(NO_TOOL_CALL_INSTRUCTION)
        assert NO_TOOL_CALL_INSTRUCTION not in messages[-1]["content"]

    @pytest.mark.parametrize(
        "raw",
        [
            "ChattyAgent",
            '"ChattyAgent"',
            "'ChattyAgent'",
            "TargetAgent('ChattyAgent')",
            'TargetAgent("ChattyAgent")',
            "TargetAgent.ChattyAgent",
            "'TargetAgent.ChattyAgent'",
            "ChattyAgent(policy_number=None)",
            "The answer is ChattyAgent.",
        ],
    )
    def test_enum_field_decorations_are_stripped(self, raw):
        from enum import StrEnum

        class TargetAgent(StrEnum):
            PolicyAgent = "PolicyAgent"
            ChattyAgent = "ChattyAgent"

        signature = dspy.Signature("chat_history -> target_agent: TargetAgent")
        signature = signature.with_updated_fields("target_agent", annotation=TargetAgent)

        result = LenientChatAdapter().parse(
            signature, f"[[ ## target_agent ## ]]\n\n{raw}\n\n[[ ## completed ## ]]\n"
        )

        assert result == {"target_agent": TargetAgent.ChattyAgent}

    def test_enum_field_without_any_member_is_left_for_dspy_to_reject(self):
        from enum import StrEnum

        class TargetAgent(StrEnum):
            PolicyAgent = "PolicyAgent"
            ChattyAgent = "ChattyAgent"

        signature = dspy.Signature("chat_history -> target_agent: TargetAgent")
        signature = signature.with_updated_fields("target_agent", annotation=TargetAgent)

        with pytest.raises(dspy.utils.exceptions.AdapterParseError):
            LenientChatAdapter().parse(
                signature, "[[ ## target_agent ## ]]\nSomethingElse\n[[ ## completed ## ]]"
            )

    def test_fallback_disabled_by_default(self):
        assert LenientChatAdapter().use_json_adapter_fallback is False

    def test_class_name_is_accepted_by_stream_listener(self):
        # dspy.streaming.StreamListener dispatches on the adapter class name.
        assert LenientChatAdapter.__name__ == "ChatAdapter"
        listener = dspy.streaming.StreamListener(signature_field_name="response")
        assert LenientChatAdapter.__name__ in listener.adapter_identifiers
