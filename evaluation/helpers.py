import helpers
import numpy as onp
from collections import Counter
import matplotlib.pyplot as plt


def get_screwnet_joint_type(qs, ds, threshold=1e-5):
    if onp.all(qs < threshold) and onp.all(ds < threshold):
        return helpers.MotionType.RIGID
    elif onp.all(qs >= threshold) and onp.all(ds < threshold):
        return helpers.MotionType.ROT
    elif onp.all(qs < threshold) and onp.all(ds >= threshold):
        return helpers.MotionType.TRANS
    else:
        return helpers.MotionType.HELIC


def get_screwnet_joint_types(q_vals, d_vals, threshold=1e-5):
    return [
        get_screwnet_joint_type(qs, ds, threshold=threshold)
        for qs, ds in zip(q_vals, d_vals)
    ]


def get_screwnet_motion_types(screwnet_data, threshold=1e-5):
    q_vals_gt = screwnet_data["labels"][..., 6]
    d_vals_gt = screwnet_data["labels"][..., 7]
    motion_types_gt = get_screwnet_joint_types(
        q_vals_gt, d_vals_gt, threshold=threshold
    )

    q_vals_pred = screwnet_data["predictions"][..., 6]
    d_vals_pred = screwnet_data["predictions"][..., 7]
    motion_types_pred = get_screwnet_joint_types(
        q_vals_pred, d_vals_pred, threshold=threshold
    )

    return motion_types_gt, motion_types_pred


def plot_decoupled_parameter_distribution(experiment, eval_results_name):
    if "DecoupledTwist" not in eval_results_name:
        return None

    parameters = experiment.read_metadata(eval_results_name + "_raw_parameters")
    portions_gt = onp.array(parameters["gt_fg"])[..., 6:8]
    portions_pred = onp.array(parameters["pred_fg"])[..., 6:8]

    bins = 20
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    # axs.hist([portions_gt[:, 0], portions_gt[:, 1]], bins=bins)
    axs.hist([portions_pred[:, 0], portions_pred[:, 1]], bins=bins)
    # ax.scatter(portions_pred[:, 0], portions_pred[:, 1], alpha=0.5)
    axs.legend(["Rotation Portion", "Translation Portion"])

    threshold = 1e-1
    motion_types_pred = [
        helpers.get_motion_type_from_decoupled_parameters(
            parameter, threshold=threshold
        )
        for parameter in parameters["pred_fg"]
    ]
    print(Counter(motion_types_pred))

    return fig
