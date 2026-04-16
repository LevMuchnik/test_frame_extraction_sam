"""Timing, metrics collection, and progress reporting."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class PhaseMetrics:
    """Metrics for a single pipeline phase."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return self.end_time - self.start_time

    def inc(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def set(self, key: str, value: float) -> None:
        self.values[key] = value

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elapsed_seconds": round(self.elapsed, 2),
            "counters": self.counters,
            "values": {k: round(v, 4) for k, v in self.values.items()},
        }


@dataclass
class PipelineMetrics:
    """Aggregate metrics for the full pipeline run."""

    video: str = ""
    phases: list[PhaseMetrics] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_elapsed(self) -> float:
        return self.end_time - self.start_time

    @contextmanager
    def phase(self, name: str):
        """Context manager that times a phase and collects its metrics."""
        pm = PhaseMetrics(name=name, start_time=time.time())
        self.phases.append(pm)
        console.print(f"\n[bold cyan]▶ Phase: {name}[/bold cyan]")
        try:
            yield pm
        finally:
            pm.end_time = time.time()
            console.print(
                f"[green]✓ {name}[/green] completed in {pm.elapsed:.1f}s"
            )

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "phases": {p.name: p.to_dict() for p in self.phases},
        }

    def save(self, path: Path) -> None:
        """Write metrics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        console.print(f"[dim]Metrics saved to {path}[/dim]")

    def print_summary(self) -> None:
        """Print a rich summary table."""
        table = Table(title=f"Pipeline Summary: {self.video}")
        table.add_column("Phase", style="cyan")
        table.add_column("Time (s)", justify="right")
        table.add_column("Key Metrics")

        for p in self.phases:
            metrics_str = ", ".join(
                [f"{k}={v}" for k, v in p.counters.items()]
                + [f"{k}={v:.3f}" for k, v in p.values.items()]
            )
            table.add_row(p.name, f"{p.elapsed:.1f}", metrics_str)

        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{self.total_elapsed:.1f}[/bold]",
            "",
        )
        console.print(table)
