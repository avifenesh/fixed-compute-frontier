"""Shared session format for the S1 state-tracking sets (see results/negeig-27b/DATA.md).

A domain provides a simulator class and a checker:

  class Sim:
      SYSTEM = {"en": "...", "he": "..."}          # what is tracked, how to answer
      def __init__(self, rng, lang, split): ...     # draw entities from the split's pools
      def initial(self) -> list[str]               # lines stating the full initial state
      def event(self) -> str                       # apply one event to the state, return its text
      def ask(self) -> tuple[str, str, dict]       # question text, exact answer, metadata

  def replay(session) -> list[str]                 # recompute every answer from the rendered text ONLY

make_session() drives a Sim into a Session: a first user turn with the initial state, then turns of 1 to
max_batch events each ending in one question, until n_events events are delivered. verify() runs the domain's
replay() and fails on any mismatch, so a generator bug cannot reach the training set.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field


@dataclass
class Turn:
    events: list[str]
    question: str
    answer: str
    meta: dict = field(default_factory=dict)


@dataclass
class Session:
    domain: str
    lang: str
    split: str
    seed: int
    n_events: int
    system: str
    initial: list[str]
    turns: list[Turn]
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Session":
        d = json.loads(s)
        d["turns"] = [Turn(**t) for t in d["turns"]]
        return Session(**d)


INITIAL_HEADER = {"en": "Initial state:", "he": "מצב התחלתי:"}
EVENTS_HEADER = {"en": "Events:", "he": "אירועים:"}


def user_text(lang: str, turn: Turn, initial: list[str] | None = None) -> str:
    parts = []
    if initial is not None:
        parts.append(INITIAL_HEADER[lang] + "\n" + "\n".join(initial))
    if turn.events:
        parts.append(EVENTS_HEADER[lang] + "\n" + "\n".join(turn.events))
    parts.append(turn.question)
    return "\n\n".join(parts)


def to_messages(s: Session) -> list[dict]:
    msgs = [{"role": "system", "content": s.system}]
    for i, t in enumerate(s.turns):
        msgs.append({"role": "user", "content": user_text(s.lang, t, s.initial if i == 0 else None)})
        msgs.append({"role": "assistant", "content": t.answer})
    return msgs


def make_session(sim_cls, domain: str, seed: int, n_events: int, lang: str, split: str,
                 max_batch: int = 16) -> Session:
    rng = random.Random(seed)
    sim = sim_cls(rng, lang, split)
    initial = sim.initial()
    turns: list[Turn] = []
    q, a, m = sim.ask()
    turns.append(Turn(events=[], question=q, answer=a, meta={**m, "after_events": 0}))
    done = 0
    while done < n_events:
        k = min(n_events - done, rng.randint(1, max_batch))
        evs = [sim.event() for _ in range(k)]
        done += k
        q, a, m = sim.ask()
        turns.append(Turn(events=evs, question=q, answer=a, meta={**m, "after_events": done}))
    return Session(domain=domain, lang=lang, split=split, seed=seed, n_events=n_events,
                   system=sim_cls.SYSTEM[lang], initial=initial, turns=turns)


def verify(session: Session, replay) -> None:
    got = replay(session)
    want = [t.answer for t in session.turns]
    if got != want:
        bad = next(i for i, (g, w) in enumerate(zip(got, want)) if g != w) if len(got) == len(want) else -1
        raise AssertionError(f"{session.domain} seed {session.seed}: replay disagrees at turn {bad}: "
                             f"{got[bad] if bad >= 0 else len(got)!r} != {want[bad] if bad >= 0 else len(want)!r}")


def split_pool(pool: list, split: str) -> list:
    """Disjoint entity pools: train takes even positions, eval odd positions."""
    return pool[0::2] if split == "train" else pool[1::2]
