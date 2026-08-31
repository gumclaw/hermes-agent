"""Tests for the reserved interactive executor lane
(``gateway.interactive_executor_workers``).

Behavior contract:
  * Unset / 0 / invalid -> lane OFF: ``_get_interactive_executor()`` returns
    the SAME object as ``_get_executor()`` (identity) — byte-identical
    behavior to a single shared pool.
  * Positive int -> a separate reserved pool of that size with the
    ``hermes-gw-interactive`` thread prefix; the shared pool is untouched.
  * ``_is_batch_platform`` routes webhook to the shared pool; interactive
    platforms and malformed sources are handled fail-safe.
  * Shutdown closes BOTH pools; further executor requests raise.
  * Starvation proof: with the shared pool fully occupied by blocked batch
    work, an interactive submission still runs immediately.
"""
import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest


def _make_runner(interactive_workers=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        interactive_executor_workers=interactive_workers
    )
    runner._executor = None
    runner._executor_lock = threading.Lock()
    runner._executor_closing = False
    return runner


def _cleanup(runner):
    for attr in ("_executor", "_interactive_executor"):
        pool = getattr(runner, attr, None)
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------- lane off

@pytest.mark.parametrize("value", [None, 0])
def test_lane_off_returns_shared_pool(value):
    runner = _make_runner(interactive_workers=value)
    try:
        assert runner._get_interactive_executor() is runner._get_executor()
    finally:
        _cleanup(runner)


def test_lane_off_when_config_attribute_missing():
    runner = _make_runner()
    runner.config = SimpleNamespace()  # old config object without the field
    try:
        assert runner._get_interactive_executor() is runner._get_executor()
    finally:
        _cleanup(runner)


def test_lane_off_when_runner_has_no_config_object():
    runner = _make_runner()
    del runner.config
    try:
        assert runner._get_interactive_executor() is runner._get_executor()
    finally:
        _cleanup(runner)


# ----------------------------------------------------------------- lane on

def test_lane_on_creates_separate_pool():
    runner = _make_runner(interactive_workers=3)
    try:
        shared = runner._get_executor()
        lane = runner._get_interactive_executor()
        assert lane is not shared
        assert lane._max_workers == 3
        assert lane._thread_name_prefix == "hermes-gw-interactive"
        assert runner._get_interactive_executor() is lane  # memoized
    finally:
        _cleanup(runner)


def test_lane_recreated_after_external_shutdown():
    runner = _make_runner(interactive_workers=2)
    try:
        lane = runner._get_interactive_executor()
        lane.shutdown(wait=False)
        lane2 = runner._get_interactive_executor()
        assert lane2 is not lane
        assert lane2.submit(lambda: "alive").result(timeout=5) == "alive"
    finally:
        _cleanup(runner)


# ----------------------------------------------------------------- routing

def _src(platform):
    return SimpleNamespace(platform=SimpleNamespace(value=platform))


def test_platform_routing():
    from gateway.run import GatewayRunner
    for p in (
        "webhook", "api_server", "msgraph_webhook", "wecom_callback",
        "whatsapp_cloud", "sms", "email", "homeassistant", "relay", "unknown",
    ):
        assert GatewayRunner._is_batch_platform(_src(p)) is True
    for p in (
        "local", "telegram", "discord", "slack", "whatsapp", "signal",
        "mattermost", "matrix", "dingtalk", "wecom", "weixin", "qqbot",
        "yuanbao",
    ):
        assert GatewayRunner._is_batch_platform(_src(p)) is False
    # plain-string platform attribute (no .value)
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform="telegram")) is False
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform="webhook")) is True
    # malformed sources fail SAFE -> batch (shared pool, old behavior)
    assert GatewayRunner._is_batch_platform(None) is True
    assert GatewayRunner._is_batch_platform(SimpleNamespace()) is True
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform=None)) is True
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform=SimpleNamespace(value=None))) is True
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform="")) is True
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform=42)) is True


def test_real_platform_enum_routing():
    from gateway.run import GatewayRunner
    from gateway.platforms.base import Platform
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform=Platform.WEBHOOK)) is True
    assert GatewayRunner._is_batch_platform(
        SimpleNamespace(platform=Platform.TELEGRAM)) is False


def test_run_agent_inner_call_site_routes_by_platform():
    """Wiring contract: the agent-turn executor task derives ``_interactive``
    from the turn's source. If the routing is deleted or inverted at the
    ``_run_agent_inner`` submission site, this fails."""
    import inspect
    from gateway.run import GatewayRunner

    src = inspect.getsource(GatewayRunner._run_agent_inner)
    assert "_interactive=not self._is_batch_platform(source)" in src


# ---------------------------------------------------------------- shutdown

def test_shutdown_closes_both_pools():
    runner = _make_runner(interactive_workers=2)
    shared = runner._get_executor()
    lane = runner._get_interactive_executor()
    assert lane is not shared
    runner._shutdown_executor()
    assert shared._shutdown
    assert lane._shutdown
    with pytest.raises(RuntimeError):
        runner._get_executor()
    with pytest.raises(RuntimeError):
        runner._get_interactive_executor()


def test_shutdown_with_lane_never_created():
    runner = _make_runner(interactive_workers=2)
    shared = runner._get_executor()
    runner._shutdown_executor()  # must not raise on missing lane
    assert shared._shutdown


# ------------------------------------------------------- starvation proof

def test_interactive_lane_immune_to_batch_saturation():
    """With the shared pool 100% blocked by batch work, an interactive task
    still completes promptly through the reserved lane."""
    runner = _make_runner(interactive_workers=2)
    try:
        shared = runner._get_executor()
        lane = runner._get_interactive_executor()

        release = threading.Event()
        blockers = [shared.submit(release.wait) for _ in range(12)]

        t0 = time.monotonic()
        result = lane.submit(lambda: "fast").result(timeout=5)
        elapsed = time.monotonic() - t0

        assert result == "fast"
        assert elapsed < 2.0, f"interactive task queued {elapsed:.1f}s behind batch"
        release.set()
        concurrent.futures.wait(blockers, timeout=5)
    finally:
        _cleanup(runner)


# --------------------------------------------- end-to-end coroutine wiring

def test_run_in_executor_with_context_routes_by_flag():
    import asyncio

    runner = _make_runner(interactive_workers=1)
    try:
        def thread_name():
            return threading.current_thread().name

        async def go():
            batch = await runner._run_in_executor_with_context(thread_name)
            inter = await runner._run_in_executor_with_context(
                thread_name, _interactive=True
            )
            return batch, inter

        batch_name, inter_name = asyncio.run(go())
        assert batch_name.startswith("hermes-gateway")
        assert inter_name.startswith("hermes-gw-interactive")
    finally:
        _cleanup(runner)
