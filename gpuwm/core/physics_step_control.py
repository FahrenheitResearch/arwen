"""Ephemeral physics decisions imposed by the domain's current step.

These scalars travel to reused tile buffers before compute. They are derived
again by the clock after restart, rather than stored as another clock or as
part of the forecast's configured identity.
"""
from dataclasses import dataclass


STEP_CONTROL_ATTRIBUTES = (
    "stepra", "stepcu", "stepbl",
    "radt_seconds", "cudt_seconds", "bldt_seconds",
    "radiation_due_override", "cumulus_due_override",
)


@dataclass(frozen=True)
class PhysicsStepControl:
    values: tuple[tuple[str, int | float | bool | None], ...]

    @classmethod
    def from_driver(cls, driver):
        if driver is None:
            return None
        return cls(tuple((name, getattr(driver, name))
                         for name in STEP_CONTROL_ATTRIBUTES
                         if hasattr(driver, name)))

    def apply(self, state) -> None:
        driver = getattr(state, "physics", None)
        if driver is None:
            if self.values:
                raise ValueError("physics step control needs a tile physics driver")
            return
        for name, value in self.values:
            setattr(driver, name, value)
