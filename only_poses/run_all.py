import helpers
import numpy as onp
import enum


class Methods(enum.Enum):
    Sturm = enum.auto()
    SturmOriginal = enum.auto()
    FactorGraph = enum.auto()
    FactorGraphGT = enum.auto()


number_samples = 50

parameter_set_rot = {
    "stddev_pos": onp.array([0.001, 0.03, 0.1]),
    "stddev_ori": onp.array([1.0, 3.0, 10.0]) * onp.pi / 180,
    "observation_amount": onp.array([5, 10, 20, 40, 80, 160, 320]),
    "sample_length": onp.array([15.0, 45.0, 90.0]) * onp.pi / 180,
}

parameter_set_trans = {
    "stddev_pos": onp.array([0.001, 0.03, 0.1]),
    "stddev_ori": onp.array([1.0, 3.0, 10.0]) * onp.pi / 180,
    "observation_amount": onp.array([5, 10, 20, 40, 80, 160, 320]),
    "sample_length": onp.array([0.05, 0.20, 0.40]),
}


methods_to_use = [Methods.FactorGraph]
methods_to_use = [Methods.FactorGraph, Methods.SturmOriginal]


def dispatch(motion_type, parameter_set):
    for pi in helpers.dict_product(parameter_set):
        sd_p = pi["stddev_pos"]
        sd_o = pi["stddev_ori"]
        ob_a = pi["observation_amount"]
        sa_l = pi["sample_length"]
        mt = motion_type

        experiment_name = (
            f"op_sd_p_{sd_p}_sd_o_{sd_o}_ob_a_{ob_a}_sa_l_{sa_l}_mt_{motion_type}"
        )
        cmd = (
            "python -m only_poses.main"
            " --experiment-root-path"
            f" ./experiments/only_poses/ --experiment-name"
            f" {experiment_name} --motion-type {mt} --stddev-pos {sd_p} --stddev-ori"
            f" {sd_o} --observation-amount {ob_a} --sample-length"
            f" {sa_l} --number-samples {number_samples} --create-samples"
        )

        # Add huber
        # cmd += " --all-hubers"
        # cmd += " --huber-delta 1"

        if not Methods.Sturm in methods_to_use:
            cmd += " --no-use-sturm"
        if not Methods.SturmOriginal in methods_to_use:
            cmd += " --no-use-sturm-original"
        if not Methods.FactorGraph in methods_to_use:
            cmd += " --no-use-fg"
        if not Methods.FactorGraphGT in methods_to_use:
            cmd += " --no-use-fg-gt"

        print(cmd)


dispatch("TRANS", parameter_set_trans)
dispatch("ROT", parameter_set_rot)
