import dcargs

import dataclasses
from typing import List, Optional

import helpers


@dataclasses.dataclass
class FullPoseConfig:
    experiment_root_path: str
    experiment_name: str

    # Number of samples to use
    number_samples: int

    # Amount of observations we have
    observation_amount: int
    # Sample step length
    sample_length: float
    # Standard deviations for position optimization and noise
    stddev_pos: float
    # Standard deviations for position optimization and noise
    stddev_ori: float
    # Joint type
    motion_type: helpers.MotionType

    # create new samples
    create_samples: bool = False

    # max restarts for each method
    max_restarts: int = 10

    # Use huber for all nodes
    all_hubers: bool = False
    # Huber delta
    huber_delta: Optional[float] = None

    # Fixed Seed
    seed: int = 1234

    use_sturm: bool = True
    use_sturm_original: bool = True
    use_fg: bool = True
    use_fg_gt: bool = True

    # Actitvate debug
    DEBUG: bool = False
    # Activate verbose output
    verbose: bool = False

    # Use the old sampling method (non-zero centered poses)
    old_method: bool = False
    # Sample noise (for old sampling method)
    variance: float = 0.1


if __name__ == "__main__":
    args = dcargs.parse(FullPoseConfig, description=__doc__)
    print(args)
