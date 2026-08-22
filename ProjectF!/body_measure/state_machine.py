"""Gesture-controlled measurement workflow."""

from enum import Enum, auto


class State(Enum):
    WAITING = auto()
    MEASURING = auto()
    EXIT = auto()


class GestureStateMachine:
    """A peace sign must be held continuously before a state transition."""

    def __init__(self, hold_seconds: float):
        self.state = State.WAITING
        self.hold_seconds = hold_seconds
        self._gesture_started: float | None = None
        self._awaiting_release = False

    def reset(self) -> None:
        """Prepare a fresh measurement cycle after a result was acknowledged."""
        self.state = State.WAITING
        self._gesture_started = None
        self._awaiting_release = False

    def update(self, peace_sign: bool, timestamp: float) -> tuple[State, float, bool]:
        if not peace_sign:
            self._gesture_started = None
            self._awaiting_release = False
            return self.state, 0.0, False
        # One continuous hold is one command.  Requiring a release prevents a
        # long start gesture from being read as a second "exit" gesture before
        # the measurement has had time to complete.
        if self._awaiting_release:
            return self.state, 1.0, False
        if self._gesture_started is None:
            self._gesture_started = timestamp
        progress = min(1.0, (timestamp - self._gesture_started) / self.hold_seconds)
        if progress < 1.0:
            return self.state, progress, False
        self._gesture_started = None
        if self.state is State.WAITING:
            self.state = State.MEASURING
        elif self.state is State.MEASURING:
            self.state = State.EXIT
        self._awaiting_release = True
        return self.state, 1.0, True
