from dataclasses import replace
from sentinal.core.execution.context import ExecutionContext


def test_context_session_defaults_none():
    ctx = ExecutionContext(input={"q": 1})
    assert ctx.session is None


def test_context_session_via_replace():
    ctx = ExecutionContext(input={"q": 1})
    sentinel = object()
    ctx2 = replace(ctx, session=sentinel)
    assert ctx2.session is sentinel
    assert ctx.session is None          # original unchanged (frozen)
    assert ctx2.input == {"q": 1}       # other fields preserved
