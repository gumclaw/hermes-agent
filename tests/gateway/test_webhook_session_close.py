"""Invariant test: a completed webhook delivery closes its session.

Regression guard for the ghost-session leak.  Webhook deliveries create a
unique one-shot session (``delivery_id`` baked into the session key), but the
adapter historically fired ``handle_message`` without ever ending the session.
``SessionDB.prune_sessions`` only reaps rows where ``ended_at IS NOT NULL``, so
every webhook session stayed unprunable and state.db grew without bound (this
was the primary driver of the SQLite lock-contention gateway outage).

The invariant asserted here is a *behavior contract*, not a snapshot: once a
webhook delivery's agent run completes, the session row for that delivery must
have ``ended_at`` set — mirroring how a cron run closes its session with
``end_session(..., "cron_complete")``.

CRITICAL: these tests go through the REAL ``handle_message`` →
``_process_message_background`` → ``on_processing_complete`` pipeline (only the
runner-side ``_message_handler`` is stubbed, exactly the seam the live gateway
injects).  ``handle_message`` is fire-and-forget — it spawns the background
task and returns before the run starts — so any close bolted around
``handle_message`` itself runs BEFORE the session row exists and silently
no-ops.  A test that fakes ``handle_message`` to create the row synchronously
masks exactly that bug (the first version of this fix shipped that way).
"""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.session import SessionSource, SessionStore


def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


class _FakeRunner:
    """Minimal gateway runner surface the webhook close path depends on.

    Wires a real ``SessionStore`` (which owns a real ``SessionDB``) and reuses
    that same ``SessionDB`` as ``_session_db`` so the row created at routing
    time is the row the close path ends — exactly the wiring the live gateway
    has (``self.session_store`` + ``self._session_db``).
    """

    def __init__(self, store: SessionStore):
        self.session_store = store
        self._session_db = store._db

    def _session_key_for_source(self, source: SessionSource) -> str:
        return self.session_store._generate_session_key(source)


def _make_store(tmp_path) -> SessionStore:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config = GatewayConfig(
        platforms={Platform.WEBHOOK: PlatformConfig(enabled=True)}
    )
    store = SessionStore(sessions_dir=sessions_dir, config=config)
    assert store._db is not None, "test requires a real SessionDB"
    return store


def _make_event(adapter: WebhookAdapter, delivery_id: str, text: str) -> MessageEvent:
    session_chat_id = f"webhook:alerts:{delivery_id}"
    source = adapter.build_source(
        chat_id=session_chat_id,
        chat_name="webhook/alerts",
        chat_type="webhook",
        user_id="webhook:alerts",
        user_name="alerts",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"message": text},
        message_id=delivery_id,
    )


async def _drain_background_tasks(adapter: WebhookAdapter, timeout: float = 5.0) -> None:
    """Wait for the adapter's spawned processing task(s) to finish."""
    deadline = asyncio.get_event_loop().time() + timeout
    while adapter._background_tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    # One extra tick for done-callbacks to run.
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_completed_webhook_delivery_closes_its_session(tmp_path):
    """After a webhook run finishes (REAL dispatch path), ended_at is set."""
    store = _make_store(tmp_path)
    runner = _FakeRunner(store)

    adapter = _make_adapter(
        {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "log",
            }
        }
    )
    adapter.gateway_runner = runner

    # Stub the RUNNER-side handler (the seam the live gateway injects) — the
    # adapter's own handle_message / _process_message_background pipeline runs
    # for real, including the fire-and-forget task spawn and the
    # on_processing_complete hook.  The handler creates the session row, just
    # like GatewayRunner._handle_message does at routing time.
    created = {}

    async def _message_handler(event: MessageEvent):
        entry = store.get_or_create_session(event.source)
        created["session_id"] = entry.session_id
        return ""  # webhook deliver=log — nothing to send back

    adapter._message_handler = _message_handler

    event = _make_event(adapter, "alert-close-001", "Alert: server on fire")

    # Exactly what _handle_webhook schedules.
    await adapter.handle_message(event)
    # handle_message is fire-and-forget: the session must NOT be expected to
    # exist yet.  (Guards against reintroducing a close wrapped around
    # handle_message itself, which ran before the row existed and no-op'd.)
    await _drain_background_tasks(adapter)

    session_id = created["session_id"]
    row = store._db.get_session(session_id)
    assert row is not None

    # INVARIANT: a completed webhook session must be closed so prune can reap it.
    assert row["ended_at"] is not None, (
        "webhook session was never closed — ended_at is NULL, so "
        "prune_sessions can never reap it (the ghost-session leak)"
    )
    assert row["end_reason"] == "webhook_complete"

    # And the closed row is actually prunable, unlike the pre-fix leak.
    pruned = store._db.prune_sessions(older_than_days=0, source="webhook")
    assert pruned >= 1
    store._db.close()


@pytest.mark.asyncio
async def test_webhook_session_closed_even_when_agent_run_raises(tmp_path):
    """A failing agent run still closes the session (FAILURE hook path)."""
    store = _make_store(tmp_path)
    runner = _FakeRunner(store)

    adapter = _make_adapter(
        {"alerts": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "deliver": "log"}}
    )
    adapter.gateway_runner = runner

    created = {}

    async def _boom(event: MessageEvent):
        # Row exists (routing happened) before the run blows up mid-turn.
        entry = store.get_or_create_session(event.source)
        created["session_id"] = entry.session_id
        raise RuntimeError("agent exploded mid-run")

    adapter._message_handler = _boom

    event = _make_event(adapter, "alert-fail-001", "x")

    await adapter.handle_message(event)
    await _drain_background_tasks(adapter)

    row = store._db.get_session(created["session_id"])
    assert row is not None
    assert row["ended_at"] is not None, (
        "session left open after a failed webhook run — the leak persists "
        "on the error path"
    )
    assert row["end_reason"] == "webhook_complete"
    store._db.close()


@pytest.mark.asyncio
async def test_end_webhook_session_awaits_async_session_db(tmp_path):
    """The close path handles the gateway's real AsyncSessionDB facade."""
    from hermes_state import AsyncSessionDB

    store = _make_store(tmp_path)
    runner = _FakeRunner(store)
    runner._session_db = AsyncSessionDB(store._db)

    adapter = _make_adapter(
        {"alerts": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "deliver": "log"}}
    )
    adapter.gateway_runner = runner

    event = _make_event(adapter, "alert-async-001", "x")
    entry = store.get_or_create_session(event.source)

    await adapter._end_webhook_session(event, event.source.chat_id)

    row = store._db.get_session(entry.session_id)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "webhook_complete"
    store._db.close()


@pytest.mark.asyncio
async def test_restart_cancelled_webhook_run_stays_recoverable(tmp_path):
    """A run cancelled by a restart drain must NOT be marked complete.

    Counterpart invariant to the ghost-leak tests above, and the reason the
    CANCELLED outcome is carved out.  A gateway restart drains in-flight work
    via ``cancel_background_tasks()``, so a webhook run that was interrupted
    mid-investigation reaches ``on_processing_complete`` as CANCELLED.  If we
    stamp ``webhook_complete`` on it, the row falls outside the recoverable set
    in ``find_latest_gateway_session_for_peer`` (live / ``agent_close`` /
    ``ws_orphan_reap``) and the restart's auto-resume pass can no longer find
    the interrupted transcript — it silently starts a fresh empty session
    instead and the real work is abandoned.

    Asserted as a behavior contract on the recovery query itself, not on the
    end_reason string, so the test still holds if the reason vocabulary
    changes.  Goes through the REAL cancel path (``cancel_background_tasks``),
    not a hand-called hook, so it also covers the drain wiring.
    """
    store = _make_store(tmp_path)
    runner = _FakeRunner(store)

    adapter = _make_adapter(
        {"alerts": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "deliver": "log"}}
    )
    adapter.gateway_runner = runner

    created = {}
    started = asyncio.Event()

    async def _interrupted_by_restart(event: MessageEvent):
        # Routing already created the row, and the run has produced real work
        # (a transcript) before the restart lands on it.
        entry = store.get_or_create_session(event.source)
        created["session_id"] = entry.session_id
        store._db.replace_messages(
            entry.session_id, [{"role": "user", "content": "investigate ticket"}]
        )
        started.set()
        # Mid-investigation when the drain cancels us.
        await asyncio.sleep(30)
        return ""

    adapter._message_handler = _interrupted_by_restart

    event = _make_event(adapter, "alert-restart-001", "x")
    await adapter.handle_message(event)
    await asyncio.wait_for(started.wait(), timeout=5.0)

    # The real restart drain path.
    await adapter.cancel_background_tasks()
    await asyncio.sleep(0.05)

    session_id = created["session_id"]
    row = store._db.get_session(session_id)
    assert row is not None

    # INVARIANT: the interrupted row is still findable by the recovery query
    # the restart auto-resume pass uses, so the transcript can be continued.
    session_key = runner._session_key_for_source(event.source)
    recovered = store._db.find_latest_gateway_session_for_peer(
        session_key=session_key,
        source="webhook",
        user_id=event.source.user_id,
        chat_id=event.source.chat_id,
        chat_type=event.source.chat_type,
        thread_id=event.source.thread_id,
    )
    assert recovered is not None, (
        "a restart-cancelled webhook run was closed as completed, so stale-route "
        "recovery can no longer find it — the interrupted work is abandoned and "
        "the auto-resume pass will synthesize an empty session instead"
    )
    assert recovered["id"] == session_id
    store._db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True], ids=["abandoned", "reconstructed"])
async def test_cancelled_webhook_retention_without_route_revisit(tmp_path, monkeypatch, resume):
    """Cancellation preserves work before retention, but cannot leak forever."""
    from gateway.platforms.base import ProcessingOutcome

    # SessionDB's default path is captured at import time. Isolate this case
    # from the author's other sessions while retaining real SQLite imports.
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", tmp_path / "state.db")
    store = _make_store(tmp_path)
    adapter = _make_adapter(
        {"alerts": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "deliver": "log"}}
    )
    adapter.gateway_runner = _FakeRunner(store)
    # Log-only webhook delivery has no platform typing surface. Keep this
    # retention test focused on the processing task whose lifetime it owns.
    adapter.config.typing_indicator = False
    try:
        started = asyncio.Event()
        created = {}
        transcript = [
            {"role": "user", "content": "Investigate ticket #42: café\nOriginal payload", "timestamp": 1700000000.0},
            {"role": "assistant", "content": "Checked the logs; investigation unfinished.", "timestamp": 1700000001.0},
        ]

        async def interrupted(event):
            entry = store.get_or_create_session(event.source)
            created["id"] = entry.session_id
            store._db.replace_messages(entry.session_id, transcript)
            started.set()
            await asyncio.Event().wait()

        adapter._message_handler = interrupted
        event = _make_event(adapter, "retention-unique-delivery", "original payload")
        await adapter.handle_message(event)
        await asyncio.wait_for(started.wait(), timeout=5)
        await adapter.cancel_background_tasks()
        await adapter.cancel_background_tasks()  # duplicate drain is harmless
        session_id = created["id"]
        row = store._db.get_session(session_id)
        assert store._db.prune_sessions(older_than_days=30, source="webhook") == 0
        assert store.load_transcript(session_id) == transcript

        if resume:
            # A fresh routing namespace has neither the SQLite routing entry nor
            # sessions.json, but shares the real temporary HERMES_HOME/state.db.
            store._db.close()
            restart_dir = tmp_path / "restart"
            restart_dir.mkdir()
            store = _make_store(restart_dir)
            recovered = store.get_or_create_session(event.source)
            assert recovered.session_id == session_id
            assert store.load_transcript(recovered.session_id) == transcript
            assert store._db.get_session(session_id)["ended_at"] is None
            adapter.gateway_runner = _FakeRunner(store)
            # Simulate the recovered run ending at a subsequent drain.
            await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

        # No resume or expiry lookup for the abandoned one-shot routing key.
        # Advance only the retention clock; no wall-clock waiting or DB mutation.
        with monkeypatch.context() as clock:
            clock.setattr("hermes_state.time.time", lambda: row["started_at"] + 31 * 86400)
            assert store._db.prune_sessions(older_than_days=30, source="webhook") == 1
        assert store._db.get_session(session_id) is None
        assert store.load_transcript(session_id) == []
    finally:
        try:
            await adapter.cancel_background_tasks()
        finally:
            store._db.close()
