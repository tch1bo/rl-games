import datetime
from pathlib import Path
from typing import Any

import structlog
import torch
from structlog.typing import EventDict
from tqdm import tqdm


def _add_timestamp(logger, method, event_dict):
    now = datetime.datetime.now()
    event_dict["timestamp"] = (
        now.strftime("%H:%M:%S.") + f"{now.microsecond // 10000:02d}"
    )
    return event_dict


class _TqdmWriteFile:
    def write(self, msg):
        tqdm.write(msg, end="")

    def flush(self):
        pass


def _round_floats(logger: Any, method_name: str, event_dict: EventDict):
    for key, value in event_dict.items():
        if isinstance(value, float):
            event_dict[key] = f"{value:.3f}"
    return event_dict


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_timestamp,
        structlog.processors.StackInfoRenderer(),
        _round_floats,
        structlog.dev.ConsoleRenderer(sort_keys=False),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(file=_TqdmWriteFile()),  # type: ignore
    cache_logger_on_first_use=True,
)


def get_logger(**kwargs: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(**kwargs)


logger = get_logger()


def save_checkpoint(
    out_dir: Path,
    state_dict: dict[str, Any],
    *,
    step: int,
    max_checkpoints: int,
) -> None:
    out_path = out_dir / f"checkpoint_{step:08d}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(".tmp")
    torch.save(state_dict, tmp_path)
    tmp_path.rename(out_path)
    logger.info("saved checkpoint", out_path=out_path)

    if max_checkpoints >= 0:
        checkpoints = sorted(out_path.parent.glob("checkpoint_*.pt"))
        for old in checkpoints[:-max_checkpoints]:
            logger.info("deleted old checkpoint", path=old)
            old.unlink()
