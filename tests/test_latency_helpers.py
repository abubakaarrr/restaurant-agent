from app.agent.nodes.generate_response import _approx_message_tokens
from app.agent.runner import _chunk_has_tool_calls


class _Chunk:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_approx_tokens_is_local_and_positive() -> None:
    assert _approx_message_tokens("hello world") >= 1
    assert _approx_message_tokens([{"role": "user", "content": "hi"}]) >= 1


def test_chunk_tool_call_detection() -> None:
    assert _chunk_has_tool_calls(_Chunk(tool_call_chunks=[{"index": 0}]))
    assert not _chunk_has_tool_calls(_Chunk(content="Hey —"))
    assert not _chunk_has_tool_calls(None)
