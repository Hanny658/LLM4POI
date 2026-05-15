from utils.math_util import (
    cal_slot_distance,
    cal_slot_distance_batch,
    construct_slots,
    delta_t_calculate,
    ccorr,
    haversine
)
from utils.sys_util import (
    get_root_dir,
    set_logger,
    seed_torch
)
try:
    from utils.pipeline_util import (
        save_model,
        count_parameters,
        test_step
    )
except ModuleNotFoundError as exc:
    if exc.name != "metric":
        raise

    def _missing_metric_dependency(*args, **kwargs):
        raise ModuleNotFoundError(
            "Training/evaluation pipeline utilities require the optional "
            "local metric module, which is not included in this repository."
        ) from exc

    save_model = _missing_metric_dependency
    count_parameters = _missing_metric_dependency
    test_step = _missing_metric_dependency
from utils.conf_util import DictToObject, Cfg

__all__ = [
    "DictToObject",
    "Cfg",
    "cal_slot_distance",
    "cal_slot_distance_batch",
    "construct_slots",
    "delta_t_calculate",
    "ccorr",
    "haversine",
    "get_root_dir",
    "set_logger",
    "seed_torch",
    "save_model",
    "count_parameters",
    "test_step"
]
