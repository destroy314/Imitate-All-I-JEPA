from .example_task import TASK_CONFIG, augment_images
import sys
sys.path.append("jepa")

TASK_CONFIG["train"]["dataset_dir"] = "data/original/Remove_lid_atm"