"""Read-only simulator contracts used by the Fix module."""

from __future__ import annotations

from typing import Any, Protocol

from backend.fix._local_contracts import Action, Tick


class Sim(Protocol):
    """Minimal deterministic simulation interface used by ROLE 5."""

    @property
    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current simulator state."""
        ...

    def step(self) -> Tick:
        """Advance and return the next simulator tick."""
        ...

    def inject(self, name: str, **kw: Any) -> None:
        """Inject a named scenario input."""
        ...

    def apply_action(self, action: Action) -> None:
        """Apply an intervention to the simulation."""
        ...

    def fork(self) -> Sim:
        """Return an independent copy of the current simulation state."""
        ...

    def now(self) -> int:
        """Return current simulation time in minutes."""
        ...


class Ctx(Protocol):
    """Read-only context available to a module callback."""

    def history(self, zone: str, system: str, minutes: int) -> list[float]:
        """Return normalized history for the preceding number of minutes."""
        ...

    def latest(self, channel: str, zone: str) -> dict[str, Any] | None:
        """Return the latest channel payload for a zone, if present."""
        ...