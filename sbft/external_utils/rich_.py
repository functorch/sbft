"""Utilities for working with rich."""
import collections
import time
import typing as t

import attrs
import numpy as np
import pydantic as pdt
import yaml
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn
from rich.syntax import Syntax
from rich.table import Table


@attrs.define()
class Header:
    """Display a header with the experiment name and time elapsed.

    Args:
    -----
        exp_name: The name of the experiment.
        start_time: The time at which the experiment started.

    How it looks:
    -------------
    ╭────────────────────────────────────────────────────────────╮
    │ SBFT: SBFT_Experiment                   Train time: 20.91s │
    ╰────────────────────────────────────────────────────────────╯
    """

    exp_name: str
    start_time: int = attrs.field(init=False, factory=time.time)

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            f"[b]SBFT:[/b] {self.exp_name}",
            f"Train time: {self._time_elapsed()}",
        )
        return Panel(grid, style="white on blue")

    def _time_elapsed(self) -> str:
        """Return time elapsed since start of training in a human readable format."""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{elapsed:.2f}s"
        return f"{elapsed / 60:.2f}m" if elapsed < 3600 else f"{elapsed / 3600:.2f}h"


@attrs.define()
class ConfigPanel:
    """Display the experiment config in a panel.

    Args:
    -----
        config: The experiment config.

    How it looks:
    -------------
    ╭─────────────────── Config ───────────────────╮
    │    1 actor_hidden_layers:                    │
    │    2 - 256                                   │
    │    3 - 256                                   │
    │    4 actor_log_std_max: 2.0                  │
    │    5 actor_log_std_min: -5.0                 │
    │    6 batch_size: 256                         │
    │    7 buffer_size: 10000                      │
    │    8 cuda: false                             │
    │    9 cuda_deterministic: true                │
    │   10 cudnn_deterministic: true               │
    │   11 discount_factor: 0.99                   │
    │   12 entropy_coef: 0.2                       │
    │   13 env_id: Pendulum-v1                     │
    │   14 exp_name: SBFT_Experiment               │
    │   15 exploration_noise: 0.1                  │
    │   16 learning_starts: 10000                  │
    │   17 learning_timesteps: 100000              │
    │   18 noise_clip: 0.5                         │
    │   19 policy_lr: 0.0003                       │
    │   20 policy_update_freq: 2                   │
    │   21 polyak_factor: 0.005                    │
    │   22 q_lr: 0.0003                            │
    │   23 q_network_hidden_layers:                │
    │   24 - 256                                   │
    ╰──────────────────────────────────────────────╯
    """

    config: pdt.BaseModel

    def __rich__(self) -> Panel:
        return Panel(
            Syntax(yaml.dump(self.config.dict()), "yaml", line_numbers=True),
            title="Config",
            style="white on blue",
        )


@attrs.define()
class TrainingPanel:
    """Display a progress bar for training.

    Args:
    -----
        progress_bar: The progress bar to display.

    How it looks:
    -------------
    ╭───────────────────────── Training ─────────────────────────╮
    │ Random exploration ━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 │
    │ Training           ━╺━━━━━━━━━━━━━━━━━━━━━━━━   5% 0:49:41 │
    │                                                            │
    ╰────────────────────────────────────────────────────────────╯
    """

    progress_bar: Progress = attrs.field(factory=lambda: Progress(auto_refresh=False))

    def __rich__(self) -> Panel:
        return Panel(self.progress_bar, title="Training")


@attrs.define()
class ObserverPanel:
    """Display a progress bar for showing episode returns.

    Args:
    -----
        history: The history of episode returns.
        progress: Used to display live episode returns.

    How it looks:
    -------------
    ╭───────────────────────── Observer ─────────────────────────╮
    │ ⠦ 18400 --> -1028.84                                       │
    ╰────────────────────────────────────────────────────────────╯
    """

    history: collections.deque[float] = attrs.field(
        factory=lambda: collections.deque(maxlen=50)
    )
    progress: Progress = attrs.field(init=False)
    _task_id: TaskID = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[yellow]{task.fields[step]}"),
            "-->",
            TextColumn("[green]{task.fields[avg_return]:.2f}"),
        )
        self._task_id = self.progress.add_task(
            "Observer", step=0, avg_return=-float("inf")
        )

    def update(self, step: int, avg_return: float) -> None:
        self.history.append(avg_return)
        self.progress.update(
            self._task_id,
            step=step,
            # Numpy can handle deques quite well, but pyright doesn't know that, so we
            # lie to it.
            avg_return=np.mean(t.cast(list[float], self.history)),
        )

    def __rich__(self) -> Panel:
        return Panel(self.progress, title="Observer")


@attrs.define()
class EpReturnHistory:
    """Display a table of episode returns, one per epoch.

        Args:
        -----
            epoch_len: The number of steps per epoch.
            table: The table to display.
            prev_step: The prev step at which a row was added to the table.

        How it looks:
        -------------
    ╭────────────────── Episode Return History ──────────────────╮
    │ 200                                               -1187.69 │
    │ 10200                                             -1138.96 │
    │ 20200                                              -249.14 │
    │                                                            │
    ╰────────────────────────────────────────────────────────────╯
    """

    data: collections.deque[tuple[int, float]] = attrs.field(
        factory=lambda: collections.deque(maxlen=25)
    )
    prev_step: int = attrs.field(init=False, default=-1)
    epoch_len: int = attrs.field(init=False, default=10_000)

    def __rich__(self) -> Panel:
        table = Table.grid(expand=True)
        table.add_column("Step", justify="left")
        table.add_column("Avg Return", justify="right")
        table.add_row("Step", "Avg Return")
        table.add_section()
        for step, avg_return in self.data:
            table.add_row(str(step), f"{avg_return:.2f}")
        return Panel(table, title="Episode Return History")

    def update(self, step: int, avg_return: float) -> None:
        if self.prev_step == -1 or step - self.prev_step >= self.epoch_len:
            self.data.append((step, avg_return))
            self.prev_step = step


@attrs.define()
class CliUI:
    """The CLI UI.

    Args:
    -----
        exp_name: The name of the experiment.
        config: The config to display.
        layout: The layout of the UI.
        config_panel: The panel to display the config.
        training_panel: The panel to display the training progress.
        observer_panel: The panel to display the observer progress.
        ep_return_history: The panel to display the episode return history.

    How it looks:
    -------------
    ╭──────────────────────────────────────────────╮
    │ SBFT: SBFT_Experiment      Train time: 4.09m │
    ╰──────────────────────────────────────────────╯
    ╭─────────────────── Config ───────────────────╮
    │    1 actor_hidden_layers:                    │
    │    2 - 256                                   │
    │    3 - 256                                   │
    │    4 actor_log_std_max: 2.0                  │
    │    5 actor_log_std_min: -5.0                 │
    │    6 batch_size: 256                         │
    │    7 buffer_size: 10000                      │
    │    8 cuda: false                             │
    │    9 cuda_deterministic: true                │
    │   10 cudnn_deterministic: true               │
    │   11 discount_factor: 0.99                   │
    │   12 entropy_coef: 0.2                       │
    │   13 env_id: Pendulum-v1                     │
    │   14 exp_name: SBFT_Experiment               │
    │   15 exploration_noise: 0.1                  │
    │   16 learning_starts: 10000                  │
    │   17 learning_timesteps: 100000              │
    │   18 noise_clip: 0.5                         │
    │   19 policy_lr: 0.0003                       │
    │   20 policy_update_freq: 2                   │
    │   21 polyak_factor: 0.005                    │
    │   22 q_lr: 0.0003                            │
    │   23 q_network_hidden_layers:                │
    │   24 - 256                                   │
    ╰──────────────────────────────────────────────╯
    ╭────────────────── Training ──────────────────╮
    │ Random exploration ━━━━━━━━━━━━ 100% 0:00:00 │
    │ Training           ━╺━━━━━━━━━━   9% 0:42:22 │
    │                                              │
    ╰──────────────────────────────────────────────╯
    ╭────────────────── Observer ──────────────────╮
    │ ⠹ 19200 --> -773.59                          │
    ╰──────────────────────────────────────────────╯
    ╭─────────── Episode Return History ───────────╮
    │ Step                              Avg Return │
    │ 200                                 -1187.69 │
    │ 10200                               -1145.85 │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    │                                              │
    ╰──────────────────────────────────────────────╯
    """

    exp_name: str
    config: pdt.BaseModel
    layout: Layout = attrs.field(factory=lambda: Layout(name="root"))
    config_panel: ConfigPanel = attrs.field(default=None)
    training_panel: TrainingPanel = attrs.field(factory=TrainingPanel)
    observer_panel: ObserverPanel = attrs.field(factory=ObserverPanel)
    ep_return_history: EpReturnHistory = attrs.field(factory=EpReturnHistory)
    live: Live = attrs.field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.config_panel is None:
            self.config_panel = ConfigPanel(self.config)
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="config", ratio=1),
            Layout(name="training", size=5),
            Layout(name="observer", size=3),
            Layout(name="history", ratio=1),
        )
        self.layout["header"].update(Header(self.exp_name))
        self.layout["config"].update(self.config_panel)
        self.layout["training"].update(self.training_panel)
        self.layout["observer"].update(self.observer_panel)
        self.layout["history"].update(self.ep_return_history)
        self.live = Live(self.layout, refresh_per_second=1, screen=True)

    @property
    def training_progress_bar(self) -> Progress:
        return self.training_panel.progress_bar

    def track_return(self, step: int, avg_return: float) -> None:
        self.observer_panel.update(step=step, avg_return=avg_return)
        self.ep_return_history.update(step=step, avg_return=avg_return)

    def start(self) -> None:
        self.live.start()

    def stop(self) -> None:
        self.live.stop()
        console = Console()
        console.rule("Training complete")
        console.print(self.ep_return_history)
