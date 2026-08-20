"""Accelerate adapter for optional Trackio experiment logging."""

from __future__ import annotations

from typing import Any

from accelerate.tracking import GeneralTracker


class TrackioTracker(GeneralTracker):
    """Expose Trackio through the Accelerate tracker interface."""

    name = "trackio"
    requires_logging_directory = False
    main_process_only = True

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        group: str | None = None,
        space_id: str | None = None,
        resume: str = "allow",
    ) -> None:
        super().__init__()
        self.project = project
        self.run_name = run_name
        self.group = group
        self.space_id = space_id
        self.resume = resume
        self._run = None

    @property
    def tracker(self):
        return self._run

    def _start(self, config: dict[str, Any] | None = None):
        if self._run is None:
            try:
                import trackio
            except ImportError as error:
                raise ImportError(
                    "Trackio tracking requires the tracking extra: "
                    "uv sync --extra tracking"
                ) from error

            init_kwargs = {
                "project": self.project,
                "name": self.run_name,
                "group": self.group,
                "space_id": self.space_id,
                "config": config,
                "resume": self.resume,
            }
            self._run = trackio.init(
                **{
                    key: value
                    for key, value in init_kwargs.items()
                    if value is not None
                }
            )
        return self._run

    def store_init_configuration(self, values: dict[str, Any]) -> None:
        self._start(values)

    def log(
        self, values: dict[str, Any], step: int | None = None, **kwargs: Any
    ) -> None:
        self._start().log(values, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
