import dcargs
from only_poses.config import FullPoseConfig
from sample_generator import JointConnection, Sample
import experiment_files
from typing import List, Dict
import numpy as onp
import factor_graph
import helpers
from evaluation.eval import Evaluation
from baseline.joints import TwistJointParameters
import baseline


def optimize_graph(
    graph: factor_graph.graph.Graph,
    poses_named,
    variance_pos=0.005,
    variance_ori=0.02,
    verbose=False,
    max_restarts=10,
    aux_data_in={},
    use_huber=False,
):
    # Copy dict!
    aux_data = aux_data_in.copy()
    # if not "best_assignment" in aux_data.keys():
    #         aux_data["best_assignment"] = None

    pose_variance = onp.concatenate(
        (
            onp.repeat(variance_pos, repeats=3),
            onp.repeat(variance_ori, repeats=3),
        )
    )

    graph.update_poses(poses_named, pose_variance, use_huber=use_huber)
    twist, base_transform, aux_data = graph.solve_graph(
        max_restarts=max_restarts, aux_data_in=aux_data
    )
    return twist, base_transform, aux_data


def create_samples(args: FullPoseConfig):
    samples = [
        Sample.generate_random(
            observation_amount=args.observation_amount,
            sample_length=args.sample_length,
            stddev_pos=args.stddev_pos,
            stddev_ori=args.stddev_ori,
            motion_type=args.motion_type,
            old_method=args.old_method,
            variance_old=args.variance,
        )
        for _ in range(args.number_samples)
    ]
    return samples


def main(args: FullPoseConfig):
    helpers.set_random_seed(args.seed)
    # jax.config.update("jax_disable_jit", True)

    experiment: experiment_files.ExperimentFiles = experiment_files.ExperimentFiles(
        args.experiment_name, root_path=args.experiment_root_path
    )  # .assert_new()
    experiment.write_metadata("config", args)

    samples: List[Sample]

    if not args.create_samples:
        try:
            samples = experiment.read_metadata("samples")
            assert len(samples) == args.number_samples
        except FileNotFoundError:
            print(
                "I tried looking for samples, but couldn't find them and will now"
                " create new ones"
            )
            samples = create_samples(args)
    else:
        samples = create_samples(args)

    experiment.write_metadata("samples", samples)

    # Consider one joint
    structure = {
        "first_second": JointConnection(
            from_id="first", to_id="second", via_id="first_second"
        )
    }

    T = args.observation_amount

    if args.use_fg or args.use_fg_gt:
        factor_graph_options = factor_graph.graph.GraphOptions(
            observe_transformation=False,
            observe_part_poses=True,
            observe_part_pose_betweens=False,
            observe_part_centers=False,
        )
        joint_formulation = {
            "first_second": factor_graph.helpers.JointFormulation.GeneralTwist
        }

        graph: factor_graph.graph.Graph = factor_graph.graph.Graph()
        variance_exp_factor = onp.concatenate(
            (
                onp.repeat(args.stddev_pos**2, repeats=3),
                onp.repeat(args.stddev_ori**2, repeats=3),
            )
        )
        graph.build_graph(
            T,
            structure,
            factor_graph_options,
            joint_formulation,
            variance_exp_factor=variance_exp_factor,
            all_hubers=args.all_hubers,
            huber_delta=args.huber_delta,
        )

    evaluator: Evaluation = Evaluation()

    # Log lists
    aux_data_fg_gt_list = []
    aux_data_fg_pred_list = []
    aux_data_sturm_list = []
    aux_data_sturm_original_list = []
    axis_estimations_list = []
    outlier_ratios_sturm = []
    outlier_ratios_sturm_original = []

    aux_data_default = {"joint_states": None, "latent_poses": None}

    def process_sample(sample: Sample):
        axis_estimations: Dict[str, TwistJointParameters] = {
            "gt": TwistJointParameters(
                sample.gts["first"][0] @ sample.base_transforms["first_second"],
                sample.twists["first_second"],
            )
        }

        if args.use_fg_gt:
            twist_gt, transform_gt, aux_data_fg_gt = optimize_graph(
                graph,
                helpers.dict_subset(sample.gts, ["first", "second"]),
                variance_pos=1e-10,
                variance_ori=1e-10,
                max_restarts=args.max_restarts,
                use_huber=args.all_hubers,
                aux_data_in=aux_data_default,
            )

            axis_estimations["gt_fg"] = TwistJointParameters(
                helpers.mean_pose(aux_data_fg_gt["latent_poses"]["first"])
                @ transform_gt,
                twist_gt,
            )

            evaluator.evaluate_linear_metrics(
                "gt_fg",
                axis_estimations["gt"],
                axis_estimations["gt_fg"],
                sample.gts["second"],
            )
            aux_data_fg_gt_list.append(aux_data_fg_gt)

        if args.use_fg:
            twist_pred, transform_pred, aux_data_fg_pred = optimize_graph(
                graph,
                helpers.dict_subset(sample.observations, ["first", "second"]),
                variance_pos=args.stddev_pos * args.stddev_pos,
                variance_ori=args.stddev_ori * args.stddev_ori,
                max_restarts=args.max_restarts,
                use_huber=args.all_hubers,
                aux_data_in=aux_data_default,
            )

            axis_estimations["pred_fg"] = TwistJointParameters(
                helpers.mean_pose(aux_data_fg_pred["latent_poses"]["first"])
                @ transform_pred,
                twist_pred,
            )

            evaluator.evaluate_linear_metrics(
                "pred_fg",
                axis_estimations["gt"],
                axis_estimations["pred_fg"],
                sample.gts["second"],
            )
            aux_data_fg_pred_list.append(aux_data_fg_pred)

        if args.use_sturm:
            (
                base_transform_sturm,
                twist_sturm,
                aux_data_sturm,
            ) = baseline.sturm.fit_pose_trajectories(
                sample.observations["first"],
                [sample.observations["second"]],
                variance_pos=args.stddev_pos * args.stddev_pos,
                variance_ori=args.stddev_ori * args.stddev_ori,
                original=False,
                max_restarts=args.max_restarts,
                aux_data_in={**aux_data_default, "outlier_ratio": None},
            )
            outlier_ratios_sturm.append(aux_data_sturm["outlier_ratio"])

            axis_estimations["pred_sturm"] = TwistJointParameters(
                helpers.mean_pose(aux_data_sturm["latent_poses"]["first"])
                @ base_transform_sturm,
                twist_sturm,
            )

            evaluator.evaluate_linear_metrics(
                "pred_sturm",
                axis_estimations["gt"],
                axis_estimations["pred_sturm"],
                sample.gts["second"],
            )
            aux_data_sturm_list.append(aux_data_sturm)

        if args.use_sturm_original:
            (
                base_transform_sturm_original,
                twist_sturm_original,
                aux_data_sturm_original,
            ) = baseline.sturm.fit_pose_trajectories(
                sample.observations["first"],
                [sample.observations["second"]],
                variance_pos=args.stddev_pos * args.stddev_pos,
                variance_ori=args.stddev_ori * args.stddev_ori,
                original=True,
                max_restarts=args.max_restarts,
                aux_data_in={**aux_data_default, "outlier_ratio": None},
            )
            outlier_ratios_sturm_original.append(
                aux_data_sturm_original["outlier_ratio"]
            )

            axis_estimations["pred_sturm_original"] = TwistJointParameters(
                helpers.mean_pose(aux_data_sturm_original["latent_poses"]["first"])
                @ base_transform_sturm_original,
                twist_sturm_original,
            )

            evaluator.evaluate_linear_metrics(
                "pred_sturm_original",
                axis_estimations["gt"],
                axis_estimations["pred_sturm_original"],
                sample.gts["second"],
            )
            aux_data_sturm_original_list.append(aux_data_sturm_original)

        axis_estimations_list.append(axis_estimations)

    def write_results():
        experiment.write_metadata("error_dict", evaluator.get_error_dict())
        experiment.write_metadata(
            "raw_data",
            {
                "aux_data_fg_gt": aux_data_fg_gt_list,
                "aux_data_fg_pred": aux_data_fg_pred_list,
                "aux_data_sturm": aux_data_sturm_list,
                "aux_data_sturm_original": aux_data_sturm_original_list,
                "axis_estimations_list": axis_estimations_list,
            },
        )
        experiment.write_metadata(
            "outlier_ratios_sturm", onp.array(outlier_ratios_sturm)
        )
        experiment.write_metadata(
            "outlier_ratios_sturm_original", onp.array(outlier_ratios_sturm_original)
        )

    import time

    times = []
    sample: Sample
    for idx, sample in enumerate(samples):
        print(f"{idx}/{len(samples)}")
        start = time.time()

        process_sample(sample)

        duration = time.time() - start
        print(f"Took {duration}s to process sample")
        times.append(duration)

        # Write intermediate results
        if idx % 50 == 0:
            write_results()
    write_results()
    print(f"mean time: {onp.array(times).mean()}")

    for key, errors in evaluator.errors.items():
        print(f"{key}: {errors.mean(axis=0)} +- {errors.std(axis=0)}\n\t{errors = }")

    if args.use_sturm:
        outliers_sturm = onp.array(outlier_ratios_sturm)
        print(f"Calculated sturm outliers {onp.mean(outliers_sturm) = }")
    if args.use_sturm_original:
        outliers_sturm_original = onp.array(outlier_ratios_sturm_original)
        print(
            f"Calculated sturm outliers original {onp.mean(outliers_sturm_original) = }"
        )


if __name__ == "__main__":
    args = dcargs.parse(FullPoseConfig, description=__doc__)
    print(args)
    main(args)
