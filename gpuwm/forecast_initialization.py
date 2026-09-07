"""Typed initialization boundary shared by prepared-cache and external inputs.

Adapters restore validated state and initialize its physics. The forecast runner
owns clocks, workspaces, stepping, restart, health checks and output.
"""
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class DomainInitialization:
    initial_result: Any
    initialize_physics: Callable[[], Any]

    @property
    def state(self):
        return self.initial_result.state


class TreeInitialization(Protocol):
    # An adapter's actual immutable forcing inventory, used to price every
    # retained interval and any time-law evaluation storage before restore.
    lateral_boundaries: Any

    def restore_domain(self, domain, grid, bundle, *, start_time,
                       scratch_arena, dycore_state_workspace) -> DomainInitialization: ...

    def verify_inputs(self, inputs) -> None: ...

    def domain_content_sha256(self, bundle) -> str: ...

    def domain_metadata(self, bundle) -> Mapping[str, object]: ...
