"""
State machine for measurement workflow.
Defines three states: WAITING, MEASURING, EXIT.
Transitions are triggered by a peace‑sign gesture with a cooldown.
"""

from enum import Enum, auto
import time

class State(Enum):
    WAITING = auto()
    MEASURING = auto()
    EXIT = auto()

class GestureStateMachine:
    def __init__(self, cooldown: float = 1.0):
        self.state = State.WAITING
        self.last_transition = time.time()
        self.cooldown = cooldown

    def update(self, peace_sign: bool) -> State:
        """Update the state based on the peace‑sign detection.
        Returns the current state after possible transition.
        """
        now = time.time()
        if now - self.last_transition < self.cooldown:
            return self.state
        if self.state == State.WAITING and peace_sign:
            self.state = State.MEASURING
            self.last_transition = now
        elif self.state == State.MEASURING and peace_sign:
            self.state = State.EXIT
            self.last_transition = now
        return self.state
