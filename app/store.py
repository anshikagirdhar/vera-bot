"""In-memory state for the bot: versioned contexts, conversations, suppression.

Everything lives in process memory (the brief allows this). A single lock keeps
context replacement atomic when the judge pushes a higher version mid-test.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

SCOPES = ("category", "merchant", "customer", "trigger")


@dataclass
class Conversation:
    conversation_id: str
    merchant_id: str | None
    customer_id: str | None = None
    trigger_id: str | None = None
    send_as: str = "vera"
    turns: list[dict] = field(default_factory=list)   # {"role": "bot"|"merchant"|"customer", "body": str}
    status: str = "active"                            # active | waiting | ended
    auto_reply_count: int = 0
    unanswered_nudges: int = 0
    language: str | None = None                       # "en" | "hi-en"
    mode: str = "pitch"                               # pitch | action

    def bot_bodies(self) -> set[str]:
        return {t["body"].strip() for t in self.turns if t["role"] == "bot"}

    def bot_turns(self) -> int:
        return sum(1 for t in self.turns if t["role"] == "bot")


class Store:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.contexts: dict[tuple[str, str], dict[str, Any]] = {}   # (scope, id) -> {version, payload}
        self.conversations: dict[str, Conversation] = {}
        self.sent_suppression_keys: set[str] = set()
        self.merchant_optout_until: dict[str, float] = {}          # merchant_id -> epoch secs
        self.merchant_last_msgs: dict[str, list[str]] = {}         # recent inbound msgs per merchant (auto-reply detection)
        self.started = time.time()

    # ---------------- contexts ----------------
    def put_context(self, scope: str, cid: str, version: int, payload: dict) -> tuple[bool, int | None]:
        """Returns (accepted, current_version_if_rejected)."""
        with self._lock:
            cur = self.contexts.get((scope, cid))
            if cur and cur["version"] >= version:
                return False, cur["version"]
            self.contexts[(scope, cid)] = {"version": version, "payload": payload}
            return True, None

    def get(self, scope: str, cid: str | None) -> dict | None:
        if not cid:
            return None
        with self._lock:
            c = self.contexts.get((scope, cid))
            return c["payload"] if c else None

    def version(self, scope: str, cid: str | None) -> int:
        if not cid:
            return 0
        with self._lock:
            c = self.contexts.get((scope, cid))
            return c["version"] if c else 0

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in SCOPES}
        with self._lock:
            for (scope, _cid) in self.contexts:
                out[scope] = out.get(scope, 0) + 1
        return out

    # ---------------- conversations ----------------
    def conversation(self, conv_id: str, merchant_id: str | None = None,
                     customer_id: str | None = None) -> Conversation:
        with self._lock:
            conv = self.conversations.get(conv_id)
            if conv is None:
                conv = Conversation(conv_id, merchant_id, customer_id)
                self.conversations[conv_id] = conv
            if merchant_id and not conv.merchant_id:
                conv.merchant_id = merchant_id
            return conv

    def has_conversation(self, conv_id: str) -> bool:
        with self._lock:
            return conv_id in self.conversations

    def note_inbound(self, merchant_id: str | None, message: str) -> int:
        """Record an inbound message for a merchant; return how many times this exact
        (normalised) text has now been seen from them (across conversations)."""
        if not merchant_id:
            return 1
        norm = " ".join(message.lower().split())
        with self._lock:
            msgs = self.merchant_last_msgs.setdefault(merchant_id, [])
            msgs.append(norm)
            del msgs[:-20]
            return msgs.count(norm)

    # ---------------- suppression ----------------
    def opt_out(self, merchant_id: str | None, days: int = 30) -> None:
        if merchant_id:
            with self._lock:
                self.merchant_optout_until[merchant_id] = time.time() + days * 86400

    def is_opted_out(self, merchant_id: str | None) -> bool:
        if not merchant_id:
            return False
        with self._lock:
            return self.merchant_optout_until.get(merchant_id, 0) > time.time()

    def teardown(self) -> None:
        with self._lock:
            self.contexts.clear()
            self.conversations.clear()
            self.sent_suppression_keys.clear()
            self.merchant_optout_until.clear()
            self.merchant_last_msgs.clear()


STORE = Store()
