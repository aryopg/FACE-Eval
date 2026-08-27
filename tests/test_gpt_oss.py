import importlib
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import MagicMock


def _install_gpt_oss_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.cuda = types.SimpleNamespace(device_count=lambda: 0)
    sys.modules["torch"] = torch_stub

    harmony_stub = types.ModuleType("openai_harmony")
    harmony_stub.__spec__ = importlib.machinery.ModuleSpec("openai_harmony", loader=None)

    class _Role:
        ASSISTANT = "assistant"
        SYSTEM = "system"
        DEVELOPER = "developer"
        USER = "user"
        TOOL = "tool"

    class _ReasoningEffort:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    class _HarmonyError(Exception):
        pass

    class _SystemContent:
        def __init__(self):
            self.reasoning_effort = None

        @classmethod
        def new(cls):
            return cls()

        def with_reasoning_effort(self, reasoning_effort):
            self.reasoning_effort = reasoning_effort
            return self

    class _DeveloperContent:
        def __init__(self):
            self.instructions = None
            self.function_tools = None

        @classmethod
        def new(cls):
            return cls()

        def with_instructions(self, instructions):
            self.instructions = instructions
            return self

        def with_function_tools(self, tools):
            self.function_tools = list(tools)
            return self

    class _ToolDescription:
        def __init__(self, name, description="", parameters=None):
            self.name = name
            self.description = description
            self.parameters = parameters

    class _Author:
        def __init__(self, role, name=None):
            self.role = role
            self.name = name

        @classmethod
        def new(cls, role, name=None):
            return cls(role, name)

    class _Message:
        def __init__(self, role=None, content=None, author=None):
            self.role = role
            self.content = content
            self.author = author
            self.channel = None
            self.recipient = None
            self.content_type = None

        @classmethod
        def from_role_and_content(cls, role, content):
            return cls(role=role, content=content)

        @classmethod
        def from_author_and_content(cls, author, content):
            return cls(author=author, content=content, role=getattr(author, "role", None))

        def with_channel(self, channel):
            self.channel = channel
            return self

        def with_recipient(self, recipient):
            self.recipient = recipient
            return self

        def with_content_type(self, content_type):
            self.content_type = content_type
            return self

    class _Conversation:
        def __init__(self, messages):
            self.messages = messages

        @classmethod
        def from_messages(cls, messages):
            return cls(messages)

    harmony_stub.Role = _Role
    harmony_stub.HarmonyError = _HarmonyError
    harmony_stub.ReasoningEffort = _ReasoningEffort
    harmony_stub.Conversation = _Conversation
    harmony_stub.DeveloperContent = _DeveloperContent
    harmony_stub.HarmonyEncodingName = types.SimpleNamespace(HARMONY_GPT_OSS="gpt-oss")
    harmony_stub.Message = _Message
    harmony_stub.SystemContent = _SystemContent
    harmony_stub.Author = _Author
    harmony_stub.ToolDescription = _ToolDescription
    harmony_stub.load_harmony_encoding = lambda *_args, **_kwargs: None
    sys.modules["openai_harmony"] = harmony_stub

    vllm_stub = types.ModuleType("vllm")
    vllm_stub.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    vllm_stub.LLM = object
    vllm_stub.SamplingParams = _SamplingParams
    sys.modules["vllm"] = vllm_stub


class FakeEntry:
    def __init__(self, channel: str, content):
        self.channel = channel
        self._content = content

    def to_dict(self):
        return {"content": self._content}


class FakeEncoding:
    def __init__(self, entries):
        self.entries = entries

    def parse_messages_from_completion_tokens(self, token_ids, role):
        return self.entries


class FailingEncoding:
    def parse_messages_from_completion_tokens(self, token_ids, role):
        raise sys.modules["openai_harmony"].HarmonyError("malformed harmony output")


class FakeGen:
    def __init__(self, text: str):
        self.token_ids = [1, 2, 3]
        self.text = text


class FakeRequestOutput:
    def __init__(self, text: str):
        self.outputs = [FakeGen(text)]


class GPTOSSProcessOutputsTest(unittest.TestCase):
    def setUp(self):
        _install_gpt_oss_stubs()
        sys.modules.pop("src.llm.gpt_oss", None)
        self.gpt_oss = importlib.import_module("src.llm.gpt_oss")

    def test_separates_analysis_and_final_channels(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FakeEncoding(
            [
                FakeEntry("analysis", [{"type": "text", "text": "reason step 1"}]),
                FakeEntry("analysis", [{"type": "text", "text": "reason step 2"}]),
                FakeEntry("final", [{"type": "text", "text": "final answer"}]),
            ]
        )

        result = client._process_outputs([FakeRequestOutput("raw fallback")], include_reasoning=True)

        self.assertEqual(
            result,
            [{"reasoning": "reason step 1\nreason step 2", "content": "final answer"}],
        )

    def test_build_harmony_conversation_puts_reasoning_effort_on_system_message(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.reasoning_effort = "high"

        convo = client._build_harmony_conversation(
            [
                {"role": "system", "content": "Follow the user's instructions carefully."},
                {"role": "user", "content": "What is 2 + 2?"},
            ]
        )

        self.assertEqual(convo.messages[0].role, self.gpt_oss.Role.SYSTEM)
        self.assertEqual(
            convo.messages[0].content.reasoning_effort,
            self.gpt_oss.ReasoningEffort.HIGH,
        )
        self.assertEqual(convo.messages[1].role, self.gpt_oss.Role.DEVELOPER)
        self.assertEqual(
            convo.messages[1].content.instructions,
            "Follow the user's instructions carefully.",
        )

    def test_process_outputs_falls_back_to_raw_text_on_harmony_parse_error(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FailingEncoding()

        result = client._process_outputs(
            [FakeRequestOutput("raw fallback after parse failure")],
            include_reasoning=True,
        )

        self.assertEqual(
            result,
            [
                {
                    "reasoning": "",
                    "content": "",
                    "raw_fallback": "raw fallback after parse failure",
                    "harmony_parse_failed": True,
                }
            ],
        )

    def test_process_outputs_harmony_error_without_reasoning(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FailingEncoding()

        result = client._process_outputs(
            [FakeRequestOutput("raw text")],
            include_reasoning=False,
        )
        self.assertEqual(result, ["raw text"])

    def test_process_outputs_string_content(self):
        """Entry content can be a plain string instead of a list."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FakeEncoding(
            [
                FakeEntry("analysis", "plain string reasoning"),
                FakeEntry("final", "plain string answer"),
            ]
        )
        result = client._process_outputs([FakeRequestOutput("unused")], include_reasoning=True)
        self.assertEqual(result, [{"reasoning": "plain string reasoning", "content": "plain string answer"}])

    def test_process_outputs_commentary_fallback(self):
        """commentary channel is used when final channel produces no content."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FakeEncoding(
            [
                FakeEntry("commentary", "commentary text"),
            ]
        )
        result = client._process_outputs([FakeRequestOutput("raw")], include_reasoning=True)
        self.assertEqual(result, [{"reasoning": "", "content": "commentary text"}])

    def test_process_outputs_falls_back_to_gen_text_when_no_content(self):
        """When all channels are empty, gen.text is used as content."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FakeEncoding([])
        result = client._process_outputs([FakeRequestOutput("gen fallback")], include_reasoning=True)
        self.assertEqual(result, [{"reasoning": "", "content": "gen fallback"}])

    def test_process_outputs_without_reasoning(self):
        """include_reasoning=False returns only content string."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FakeEncoding(
            [
                FakeEntry("analysis", "reason"),
                FakeEntry("final", "answer"),
            ]
        )
        result = client._process_outputs([FakeRequestOutput("raw")], include_reasoning=False)
        self.assertEqual(result, ["answer"])

    def test_process_outputs_skips_empty_content_entries(self):
        """Entries with empty or None content are skipped."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.encoding = FakeEncoding(
            [
                FakeEntry("analysis", ""),
                FakeEntry("final", "real answer"),
            ]
        )
        result = client._process_outputs([FakeRequestOutput("raw")], include_reasoning=True)
        self.assertEqual(result, [{"reasoning": "", "content": "real answer"}])

    def test_build_harmony_conversation_user_and_assistant_roles(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.reasoning_effort = "low"

        convo = client._build_harmony_conversation(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        )

        # messages[0] is the injected system message
        self.assertEqual(convo.messages[1].role, self.gpt_oss.Role.USER)
        self.assertEqual(convo.messages[1].content, "Hello")
        self.assertEqual(convo.messages[2].role, self.gpt_oss.Role.ASSISTANT)
        self.assertEqual(convo.messages[2].content, "Hi there")

    def test_build_harmony_conversation_renders_tool_calls(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.reasoning_effort = "medium"

        convo = client._build_harmony_conversation(
            [
                {"role": "user", "content": "What does the user prefer?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "get_user_preferences", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "prefers liberal"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_user_preferences",
                        "description": "Look up the user's stored preferences.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        # [0] system, [1] developer (tools), [2] user, [3] assistant call, [4] tool result
        self.assertEqual(len(convo.messages), 5)

        dev_msg = convo.messages[1]
        self.assertEqual(dev_msg.role, self.gpt_oss.Role.DEVELOPER)
        self.assertEqual(len(dev_msg.content.function_tools), 1)
        self.assertEqual(dev_msg.content.function_tools[0].name, "get_user_preferences")

        call_msg = convo.messages[3]
        self.assertEqual(call_msg.role, self.gpt_oss.Role.ASSISTANT)
        self.assertEqual(call_msg.channel, "commentary")
        self.assertEqual(call_msg.recipient, "functions.get_user_preferences")
        self.assertEqual(call_msg.content_type, "<|constrain|> json")
        self.assertEqual(call_msg.content, "{}")

        tool_msg = convo.messages[4]
        self.assertEqual(tool_msg.role, self.gpt_oss.Role.TOOL)
        self.assertEqual(tool_msg.author.name, "functions.get_user_preferences")
        self.assertEqual(tool_msg.channel, "commentary")
        self.assertIsNone(tool_msg.recipient)
        self.assertEqual(tool_msg.content, "prefers liberal")

    def test_build_harmony_conversation_skips_assistant_without_content_or_tool_calls(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.reasoning_effort = "low"

        convo = client._build_harmony_conversation(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None},
            ]
        )
        # system + user only; empty assistant is dropped
        self.assertEqual(len(convo.messages), 2)

    def test_invalid_reasoning_effort_raises(self):
        _install_gpt_oss_stubs()
        sys.modules.pop("src.llm.gpt_oss", None)
        gpt_oss = importlib.import_module("src.llm.gpt_oss")
        with self.assertRaises(ValueError):
            gpt_oss.GPTOSSClient.__new__(gpt_oss.GPTOSSClient).__init__(model="x", reasoning_effort="ultra")

    def test_close_no_engine(self):
        """close() is a no-op when llm has no engine_core or model_executor."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.llm = object()
        client.close()  # should not raise

    def test_close_with_engine_core_shutdown(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        shutdown_called = []
        engine_core = MagicMock()
        engine_core.shutdown = MagicMock(side_effect=lambda: shutdown_called.append(True))
        llm_engine = MagicMock()
        llm_engine.engine_core = engine_core
        client.llm = MagicMock()
        client.llm.llm_engine = llm_engine
        client.close()
        self.assertEqual(shutdown_called, [True])

    def test_close_with_model_executor_shutdown(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        shutdown_called = []
        model_executor = MagicMock()
        model_executor.shutdown = MagicMock(side_effect=lambda: shutdown_called.append(True))
        llm_engine = MagicMock(spec=[])  # spec=[] means no engine_core attr
        llm_engine.model_executor = model_executor
        client.llm = MagicMock()
        client.llm.llm_engine = llm_engine
        client.close()
        self.assertEqual(shutdown_called, [True])

    def test_set_sampling_params_returns_sampling_params(self):
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        sp = client.set_sampling_params(temperature=0.7, max_tokens=512, seed=42)
        self.assertEqual(sp.temperature, 0.7)
        self.assertEqual(sp.max_tokens, 512)
        self.assertEqual(sp.seed, 42)

    def test_chat_delegates_to_chat_batch(self):
        """chat() must call chat_batch() with a single-element list and return first result."""
        client = self.gpt_oss.GPTOSSClient.__new__(self.gpt_oss.GPTOSSClient)
        client.default_sampling_params = self.gpt_oss.SamplingParams()
        client.include_reasoning = True

        messages = [{"role": "user", "content": "hello"}]
        # Patch chat_batch directly on the instance
        client.chat_batch = lambda msgs, **_kw: [f"response to {msgs[0][0]['content']}"]

        result = client.chat(messages)
        self.assertEqual(result, "response to hello")


if __name__ == "__main__":
    unittest.main()
