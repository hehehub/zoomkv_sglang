"""Decode only after every request in the concurrent wave has finished prefill.

SGLang removes a request from ``waiting_queue`` as soon as its prefill batch
is built. In the overlap loop that request is not in ``running_batch`` yet,
so a waiting-queue-only hold lets finished prefills start decoding while the
last 128k prefill is still running.

Rules:
- Always allow extend (prefill).
- While the waiting queue is non-empty, hold decode.
- Once any request in this wave has been parked, hold decode until
  ``parked == cohort`` (default: max_running_requests).
- Server warmup is a single short request with parked==0 and waiting==0;
  that must be allowed to decode so the server can become ready.
"""

from __future__ import annotations

import os
import time

_INSTALLED = False


def prefill_all_first_enabled() -> bool:
    return os.environ.get("SGLANG_ZOOMKV_PREFILL_ALL_FIRST", "1") not in (
        "0",
        "false",
        "False",
        "",
    )


def parked_cache_count(scheduler) -> int:
    try:
        from sglang_zoomkv.attention.cuda_graph import _zoomkv_decode_backend

        runner = scheduler.tp_worker.model_runner
        zoomkv = _zoomkv_decode_backend(getattr(runner, "attn_backend", None))
        parked = getattr(zoomkv, "_parked_caches", None) or {}
        return len(parked)
    except Exception:
        return 0


def cohort_size(scheduler) -> int:
    env = os.environ.get("SGLANG_ZOOMKV_PREFILL_COHORT", "").strip()
    if env:
        return max(1, int(env))
    return max(1, int(getattr(scheduler, "max_running_requests", 1) or 1))


def suppress_decode(
    waiting: int,
    is_extend: bool,
    *,
    parked: int = 0,
    cohort: int = 0,
    running: int = 0,
) -> bool:
    """Hold decode only for an in-progress multi-request prefill wave."""
    if is_extend:
        return False
    if waiting > 0:
        return True
    # Mid-wave: at least one prefill is parked, but the cohort is not full yet.
    # This covers the last request after it left waiting_queue and before park.
    if parked > 0 and cohort > 0 and parked < cohort:
        return True
    # parked==0 and waiting==0 is either warmup or the joint decode after the
    # sidecar attach cleared _parked_caches. Holding that path sleeps 50ms
    # per token and looks like one-by-one decode.
    return False


def _assign_capacity(obj, new_size: int) -> None:
    """Write the field even when the holder is a frozen dataclass."""
    if obj is None or not hasattr(obj, "max_total_num_tokens"):
        return
    object.__setattr__(obj, "max_total_num_tokens", int(new_size))


def sync_compacted_pool_capacity(scheduler) -> None:
    """Keep leak-check totals in sync after ZoomKV shrinks the KV pool.

    Compact drops the physical pool from 144000 to ~17k. Idle invariant
    still uses the original max_total_num_tokens, sees available=17456,
    and kills the server. That abort is what resets /server_info.
    """
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return
    new_size = int(getattr(allocator, "size", 0) or 0)
    old = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
    if new_size <= 0 or new_size >= old:
        return
    _assign_capacity(scheduler, new_size)
    for obj_name in ("invariant_checker", "pool_stats_observer", "load_inquirer"):
        _assign_capacity(getattr(scheduler, obj_name, None), new_size)
    runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    _assign_capacity(runner, new_size)
    print(
        f"[ZoomKV] synced scheduler KV capacity {old} -> {new_size}",
        flush=True,
    )


def install_prefill_all_first() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler.get_next_batch_to_run
    original_idle = Scheduler.on_idle
    if getattr(original, "_zoomkv_prefill_all_first", False) and getattr(
        original_idle, "_zoomkv_prefill_all_first", False
    ):
        _INSTALLED = True
        return

    def get_next_batch_to_run(self, running_batch, last_batch=None):
        sync_compacted_pool_capacity(self)
        cohort = cohort_size(self)
        parked = parked_cache_count(self)
        waiting = len(self.waiting_queue)
        if prefill_all_first_enabled() and (
            waiting > 0 or (parked > 0 and parked < cohort)
        ):
            running_batch.batch_is_full = False
        plan = original(self, running_batch, last_batch)
        if not prefill_all_first_enabled():
            return plan
        waiting = len(self.waiting_queue)
        parked = parked_cache_count(self)
        running_n = len(plan.running_batch.reqs)
        batch = plan.batch_to_run
        is_extend = batch is not None and batch.forward_mode.is_extend()
        if suppress_decode(
            waiting,
            is_extend,
            parked=parked,
            cohort=cohort,
            running=running_n,
        ):
            now = time.monotonic()
            last = float(getattr(self, "_zoomkv_hold_logged_at", 0.0))
            if now - last >= 5.0:
                self._zoomkv_hold_logged_at = now
                available = self.token_to_kv_pool_allocator.available_size()
                print(
                    "[ZoomKV] holding decode until every prefill finishes "
                    f"(waiting={waiting}, running={running_n}, "
                    f"parked={parked}/{cohort}, available_tokens={available})",
                    flush=True,
                )
            from sglang.srt.managers.schedule_batch import NextBatchPlan

            time.sleep(0.05)
            return NextBatchPlan(
                batch_to_run=None, running_batch=plan.running_batch
            )
        self._zoomkv_hold_logged_at = 0.0
        if waiting > 0 and running_n > 0:
            print(
                "[ZoomKV] prefill before decode "
                f"(running={running_n}, waiting={waiting}, parked={parked}/{cohort})",
                flush=True,
            )
        return plan

    if not getattr(original, "_zoomkv_prefill_all_first", False):
        get_next_batch_to_run._zoomkv_prefill_all_first = True
        Scheduler.get_next_batch_to_run = get_next_batch_to_run

    if not getattr(original_idle, "_zoomkv_prefill_all_first", False):

        def on_idle(self):
            sync_compacted_pool_capacity(self)
            return original_idle(self)

        on_idle._zoomkv_prefill_all_first = True
        Scheduler.on_idle = on_idle

    _INSTALLED = True
