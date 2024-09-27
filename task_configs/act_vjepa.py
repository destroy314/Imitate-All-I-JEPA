from .act_baseline import TASK_CONFIG, augment_images

TASK_CONFIG["train"]["obs_len"] = 16
TASK_CONFIG["common"]["policy_config"]["backbone"] = "vjepa"