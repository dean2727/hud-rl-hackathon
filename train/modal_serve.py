"""Spawn a Modal policy server and return its public tunnel address."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

SERVE_WAIT_S = 900


@contextmanager
def modal_policy_server(
    checkpoint: str,
    *,
    policy_family: str = "pi05",
    wait_s: int = SERVE_WAIT_S,
    addr_queue_name: str | None = None,
) -> Iterator[str]:
    """Yield ``host:port`` for a Modal-served pi0/pi0.5 policy; cancel on exit."""
    import modal
    from train.modal_app import SERVE_ADDR_QUEUE, app, serve_policy

    queue_name = addr_queue_name or SERVE_ADDR_QUEUE
    with app.run():
        addr_q = modal.Queue.from_name(queue_name, create_if_missing=True)
        try:
            addr_q.clear()
        except Exception:
            pass
        call = serve_policy.spawn(checkpoint=checkpoint, policy_family=policy_family)
        try:
            remote = addr_q.get(timeout=wait_s)
            if not remote:
                raise RuntimeError("Modal policy server did not publish a tunnel address")
            print(f"[modal-serve] policy at {remote} (checkpoint={checkpoint!r})", flush=True)
            yield str(remote)
        finally:
            call.cancel()


__all__ = ["SERVE_WAIT_S", "modal_policy_server"]
