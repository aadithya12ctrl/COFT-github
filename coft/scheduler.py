"""Forgetting-aware delta scheduler.

COFT trains on ``T + delta`` tokens but evaluates at ``T``. ``delta`` is not
fixed: it follows the live forgetting signal F_t,

    delta_{t+1} = min(delta_t + step, delta_max)   if F_t > F_{t-1}  (forgetting accelerating)
    delta_{t+1} = max(delta_t - step, delta_min)   if F_t < F_{t-1}  (retention stabilising)

i.e. the harder the middle of the context is, the further the model is pushed
past its inference length, and the overextension is relaxed once retention
improves.
"""

from __future__ import annotations

__all__ = ["DeltaScheduler"]


class DeltaScheduler:
    def __init__(
        self,
        delta_init: int = 128,
        delta_step: int = 64,
        delta_min: int = 0,
        delta_max: int = 256,
        tolerance: float = 0.0,
    ) -> None:
        """
        Args:
            delta_init: starting overextension, in tokens.
            delta_step: amount added/removed per update.
            delta_min / delta_max: clamp. ``T + delta_max`` must stay inside the
                model's position-embedding table (1024 for GPT-2).
            tolerance: dead zone on ``|F_t - F_{t-1}|``. With the default 0 the
                scheduler reacts to every change, which is faithful to the
                paper but jittery on small/noisy datasets; a small positive
                value (e.g. 0.01) holds delta steady through noise.
        """
        if delta_min > delta_max:
            raise ValueError("delta_min must be <= delta_max")
        self.step = int(delta_step)
        self.min = int(delta_min)
        self.max = int(delta_max)
        self.tolerance = float(tolerance)

        self.delta = int(min(max(delta_init, delta_min), delta_max))
        self.prev_ft: float | None = None
        self.history: list[int] = []
        self.ft_history: list[float] = []

    def update(self, ft: float) -> int:
        """Feed in the newest F_t and get back the delta for the next step."""
        self.ft_history.append(float(ft))
        if self.prev_ft is not None:
            change = ft - self.prev_ft
            if change > self.tolerance:
                self.delta = min(self.delta + self.step, self.max)
            elif change < -self.tolerance:
                self.delta = max(self.delta - self.step, self.min)
        self.prev_ft = float(ft)
        self.history.append(self.delta)
        return self.delta

    def state_dict(self) -> dict:
        return {
            "delta": self.delta,
            "prev_ft": self.prev_ft,
            "history": list(self.history),
            "ft_history": list(self.ft_history),
        }

    def load_state_dict(self, state: dict) -> None:
        self.delta = state["delta"]
        self.prev_ft = state["prev_ft"]
        self.history = list(state.get("history", []))
        self.ft_history = list(state.get("ft_history", []))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"DeltaScheduler(delta={self.delta}, step={self.step}, "
            f"range=[{self.min}, {self.max}])"
        )
