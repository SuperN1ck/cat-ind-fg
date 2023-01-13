import enum
import functools
from typing import Dict, List, Tuple  # Any, Callable, Optional, Tuple, NamedTuple

import helpers
import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp
import jax.scipy.optimize
import jaxlie
from baseline import joints
import sturm_articulation

from scipy import optimize

from baseline import joints

# JointTypesToUse = [joints.RigidJoint, joints.PrismaticJoint, joints.RevoluteJoint]
JointTypesToUse = [joints.PrismaticJoint, joints.RevoluteJoint]
# JointTypesToUse = [joints.RigidJoint]
# JointTypesToUse = [joints.PrismaticJoint]
# JointTypesToUse = [joints.RevoluteJoint]

ESTIMATE_OUTLIERS_DURING_OPTIM = True
PRIOR_OUTLIER_RATIO = onp.log(0.01) / (-0.05)
USE_JAX_SCIPY_MINIMIZE = False


if USE_JAX_SCIPY_MINIMIZE:
    BFGS = jax.jit(
        functools.partial(
            jax.scipy.optimize.minimize,
            method="BFGS",
            options={"maxiter": 100},
        ),
        static_argnums=(0,),
    )
else:
    BFGS = functools.partial(
        optimize.minimize,
        method="BFGS",
        options={"maxiter": 100},
    )


@functools.partial(jax.jit, static_argnums=(1,))
def calculate_residual_single_observation(
    observed_transformation,
    JointModel,
    parameters,
    variance_pos,
    variance_ori,
):
    joint_state = JointModel.backward(parameters, observed_transformation)
    estimated_transformation = JointModel.forward(parameters, joint_state)
    rel_transformation: jaxlie.SE3 = (
        observed_transformation.inverse() @ estimated_transformation
    )

    # Both are the same, but I think this one is more neat
    error_residual_mine = -0.5 * jnp.sum(
        jaxlie.manifold.rminus(estimated_transformation, observed_transformation) ** 2
        / jnp.concatenate((jnp.repeat(variance_pos, 3), jnp.repeat(variance_ori, 3)))
    )

    err_position = jnp.linalg.norm(rel_transformation.translation())
    err_orientation = jnp.linalg.norm(rel_transformation.rotation().log())
    error_residual_original = -0.5 * (
        err_position**2 / variance_pos + err_orientation**2 / variance_ori
    )

    constant = -jnp.log(2 * jnp.pi * jnp.sqrt(variance_pos) * jnp.sqrt(variance_ori))
    error_residual = error_residual_original

    ll = error_residual + constant

    # jax.experimental.host_callback.id_print(
    #     {
    #         "rel_transformation": rel_transformation,
    #         "err_position": err_position,
    #         "err_orientation": err_orientation,
    #         "error_residual_mine": error_residual_mine,
    #         "error_residual_original": error_residual_original,
    #         "ll": ll,
    #         "joint_state": joint_state,
    #         "constant": constant,
    #     }
    # )
    return ll


@jax.jit
def calculate_outlier_residual(
    variance_pos,
    variance_ori,
):
    # Magic number in code of Sturm et al.
    # chi2inv = 1.96

    #### Original Implementation
    # // outside the 95% ellipsoid
    # double chi2inv = 1.96;
    # // chi2inv = 6.63; // 99% ellipsoid
    #####
    # I am not sure where the number is coming from because
    # chi2inv = 5.9915  # chi2.ppf(0.95, df=2)
    # chi2inv = 9.2103  # chi2.ppf(0.99, df=2)
    # Hence, I would guess it's hand-tuned.
    # Unfortunately, in their report there is also no futher information.
    # 1.96 did not yield good results though and hence, we are using the actual
    # value of chi2inv(0.95, df=2)
    chi2inv = 5.9915

    err_position = chi2inv * jnp.sqrt(variance_pos)
    err_orientation = chi2inv * jnp.sqrt(variance_ori)

    error_residual = -0.5 * (
        err_position**2 / variance_pos + err_orientation**2 / variance_ori
    )
    constant = -jnp.log(2 * jnp.pi * jnp.sqrt(variance_pos) * jnp.sqrt(variance_ori))
    return error_residual + constant


@functools.partial(
    jax.jit,
    static_argnums=(
        4,
        5,
    ),
)
def calculate_residuals(
    parameters,
    stacked_datapoints: jaxlie.SE3,
    variance_pos=1.0,
    variance_ori=1.0,
    JointModel=None,
    flat_parameters=False,
):
    if flat_parameters:
        parameters = JointModel.parameter_type.from_flat_parameter_vector(parameters)

    def single_observation(observed_transformation):
        return calculate_residual_single_observation(
            observed_transformation, JointModel, parameters, variance_pos, variance_ori
        )

    residuals = jax.vmap(single_observation)(stacked_datapoints)
    return residuals


# @functools.partial(
#     jax.custom_jvp
#     @functools.partial(
#         jax.jit,
#         static_argnums=(
#             4,
#             5,
#             6,
#             7,
#             8,
#         ),
#     ),
#     nondiff_argnums=(1, 2, 3, 4, 5,6,7,8)
# )
@functools.partial(
    jax.jit,
    static_argnums=(
        4,
        5,
        6,
        7,
        8,
    ),
)
def calculate_nll(
    parameters,
    stacked_datapoints: jaxlie.SE3,
    variance_pos=1.0,
    variance_ori=1.0,
    JointModel=None,
    flat_parameters=False,
    em_iterations=10,
    estimate_outliers=False,
    return_outlier_ratio=False,
):
    log_likelihood_inlier = calculate_residuals(
        parameters,
        stacked_datapoints,
        variance_pos,
        variance_ori,
        JointModel,
        flat_parameters,
    )

    if estimate_outliers or return_outlier_ratio:
        log_likelihood_outlier = calculate_outlier_residual(variance_pos, variance_ori)

        # jax.experimental.host_callback.id_print(
        #     {
        #         "log_likelihood_inlier": log_likelihood_inlier,
        #         "log_likelihood_outlier": log_likelihood_outlier,
        #     }
        # )

        outlier_ratio = 1 / 2

        for _ in range(em_iterations):
            # E Step
            likelihoods_inlier = (1.0 - outlier_ratio) * jnp.exp(log_likelihood_inlier)
            likelihood_outlier = outlier_ratio * jnp.exp(log_likelihood_outlier)
            outlier_ratios = likelihood_outlier / (
                likelihoods_inlier + likelihood_outlier
            )

            # jax.experimental.host_callback.id_print(
            #     {
            #         "likelihoods_inlier": likelihoods_inlier,
            #         "likelihood_outlier": likelihood_outlier,
            #         "outlier_ratios": outlier_ratios,
            #     }
            # )

            outlier_ratio = jnp.mean(outlier_ratios)

        # This is the original implementation but should be
        # - a multiply imo and
        # - and scaled by the outlier ratio estimate
        # see implementation below, but this does not seem to affect perfomance
        ll_original = jnp.sum(
            jnp.log(jnp.exp(log_likelihood_inlier) + jnp.exp(log_likelihood_outlier))
        )
        ll_direct = jnp.sum(
            (1 - outlier_ratios) * log_likelihood_inlier
            + outlier_ratios * log_likelihood_outlier
        )
        ll = ll_direct
        # ll = ll_direct
        # jax.experimental.host_callback.id_print(
        #     {
        #         "ll_original": ll_original,
        #         "ll direct": ll_direct,
        #     }
        # )

        # make negative log likelihood
        final_nll = -(
            ll
            - PRIOR_OUTLIER_RATIO * outlier_ratio * stacked_datapoints.wxyz_xyz.shape[0]
        )
        if not return_outlier_ratio:
            return final_nll
        else:
            return final_nll, outlier_ratio
    else:
        # jax.experimental.host_callback.id_print(
        #     {
        #         "log_likelihood_inlier": log_likelihood_inlier,
        #     }
        # )
        return -jnp.sum(log_likelihood_inlier)


# @calculate_nll.defjvp
# def calculate_nll_jvp(primals, tangents):

partial_nlls = {
    joint_type: functools.partial(
        calculate_nll,
        JointModel=joint_type,
        flat_parameters=True,
        estimate_outliers=ESTIMATE_OUTLIERS_DURING_OPTIM,
    )
    for joint_type in JointTypesToUse
}


class ObsNNLOptimizer(enum.Enum):
    BFGS = enum.auto()
    no_optimizer = enum.auto()


def fit_pose_trajectories(
    base_poses: List[jaxlie.SE3],
    parts_poses: List[List[jaxlie.SE3]],
    variance_pos=0.005,
    variance_ori=0.02,
    aux_data_in={},
    max_restarts=10,
    original=True,
    known_jt=None,
):
    assert len(parts_poses) == 1, f"{len(parts_poses)}"
    aux_data = aux_data_in.copy()

    # TODO Make multi DoF
    for part_poses in parts_poses:
        relative_poses = [
            base_pose.inverse() @ part_pose
            # part_pose.inverse() @ base_pose
            for base_pose, part_pose in zip(base_poses, part_poses)
        ]

        if original:
            twist, base_transform, aux_data = fit_joint_original(
                relative_poses,
                variance_pos=2 * variance_pos,  # Relative poses have larger variance.
                variance_ori=2 * variance_ori,
                iterations=max_restarts,
                known_jt=known_jt,
                aux_data_in=aux_data,
            )
        else:
            twist, base_transform, aux_data = fit_joint(
                relative_poses,
                variance_pos=2 * variance_pos,
                variance_ori=2 * variance_ori,
                # optimizer=ObsNNLOptimizer.no_optimizer,
                optimizer=ObsNNLOptimizer.BFGS,
                iterations=max_restarts,
                aux_data_in=aux_data,
            )

    if "latent_poses" in aux_data.keys():
        aux_data["latent_poses"] = {"first": base_poses, "second": parts_poses[0]}

    return (base_transform, twist, aux_data)


def fit_joint_original(
    datapoints: List[jaxlie.SE3],
    variance_pos: float = 0.005,
    variance_ori: float = 360 * onp.pi / 180,
    iterations=10,
    known_jt: helpers.MotionType = None,
    aux_data_in={},
):
    aux_data = aux_data_in.copy()
    options = sturm_articulation.Parameters()
    if known_jt is not None:
        if known_jt == helpers.MotionType.ROT:
            options.check_prismatic = False
            options.check_revolute = True
        elif known_jt == helpers.MotionType.TRANS:
            options.check_prismatic = True
            options.check_revolute = False
    else:
        options.check_prismatic = True
        options.check_revolute = True

    options.check_rigid = False
    options.sigma_position = onp.sqrt(variance_pos)
    options.sigma_orientation = onp.sqrt(variance_ori)
    # Not exposed --> Should stick to 10
    # options.sac_iterations = ...
    # options.optimizer_iterations =
    print(options)

    poses = [pose.wxyz_xyz for pose in datapoints]
    results = sturm_articulation.optimize(poses, options)
    print(results)

    if results.joint_type == "prismatic":
        base_transform = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.identity(), results.rigid_position
        )
        twist = jnp.concatenate([results.prismatic_dir, jnp.zeros(3)])
    elif results.joint_type == "revolute":
        base_transform = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(results.rot_axis), results.rot_center
        )
        twist = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    if "joint_states" in aux_data.keys():
        aux_data["joint_states"] = jnp.array(results.joint_configurations)
    if "outlier_ratio" in aux_data.keys():
        aux_data["outlier_ratio"] = results.outlier_ratio

    return twist, base_transform, aux_data


def fit_joint(
    datapoints: List[jaxlie.SE3],
    visualize: bool = False,
    iterations: int = 20,
    variance_pos: float = 0.005,
    variance_ori: float = 360 * onp.pi / 180,
    optimizer: ObsNNLOptimizer = ObsNNLOptimizer.BFGS,  # ["BFGS", "no_optimizer"]
    outlier_iterations=10,
    sac_iterations=100,
    em_iterations=10,
    aux_data_in={},
) -> Tuple[jnp.ndarray, jaxlie.SE3, helpers.MotionType]:
    aux_data = aux_data_in.copy()
    ## Follow Sturm et al. 2011 --> MLESAC maximum likelihood consensus
    # 1. minimal set of randomly drawn samples --> estimate parameters
    #   1.1 compute data likelihood for whole observation
    #   1.2 pick best parameters
    # 2. refine parameter vector
    key = jax.random.PRNGKey(0)
    GOOD_ENOUGH_NLL = 1e-5  # Set very high for quicker debug

    best_parameters: joints.BaseJointParameters = None
    best_nll: float = jnp.inf
    best_joint_model: joints.BaseJoint = None
    stacked_datapoints = helpers.batch_samples(datapoints)

    for JointModel in JointTypesToUse:
        parameters = []
        nlls = []
        print("+++++++ Estimating for", JointModel.__name__, "+++++++")
        for _ in range(sac_iterations):
            # TODO For speed up?
            # samples = jax.random.choice(key, flat_datapoints, shape=[JointModel.min_samples], replace=False)
            samples = onp.random.choice(
                datapoints, size=JointModel.min_samples, replace=False
            )
            parameters_ = JointModel.estimate_parameters(samples, visualize=visualize)

            nll_ = partial_nlls[JointModel](
                parameters_.get_flat_parameter_vector(),
                stacked_datapoints,
                variance_pos,
                variance_ori,
            )

            print("negative log likelihood", nll_)

            nlls.append(nll_)
            parameters.append(parameters_)

            # if nll_ < GOOD_ENOUGH_NLL:
            #     break

            _, key = jax.random.split(key)

        # Get starting point
        nlls = jnp.asarray(nlls)
        best_fit_idx = jnp.argmin(nlls)

        final_nll = nlls[best_fit_idx]
        final_parameters = parameters[best_fit_idx]

        if final_nll > GOOD_ENOUGH_NLL or True:  # Always minimize
            x0 = parameters[best_fit_idx].get_flat_parameter_vector()
            print("Trying to minimize {} with {}".format(nlls[best_fit_idx], x0))
            # min_func = jax.jit(
            # min_func = functools.partial(
            #     calculate_nll,
            #     stacked_datapoints=stacked_datapoints,
            #     variance=variance,
            #     JointModel=JointModel,
            #     flat_parameters=True,
            # )
            # )

            if optimizer == ObsNNLOptimizer.BFGS:
                # calc_nll = jax.jit(
                #     functools.partial(
                #         calculate_nll,
                #         variance=variance,
                #         JointModel=JointModel,
                #         flat_parameters=True,
                #     )
                # )

                min_result = BFGS(
                    partial_nlls[JointModel],
                    x0,
                    args=(stacked_datapoints, variance_pos, variance_ori),
                )

                # status – integer solver specific return code.
                status_mapping = {
                    0: "converged (nominal)",
                    1: "max BFGS iters reached",
                    3: "zoom failed",
                    4: "saddle point reached",
                    5: "max line search iters reached",
                    -1: "undefined",
                }

                print(
                    f"Minimizing done...\n Success: {min_result.success}, Status"
                    f" {min_result.status}\nReached {min_result.fun} with"
                    f" {min_result.x}"
                )
                if USE_JAX_SCIPY_MINIMIZE:
                    print(f"status message: {status_mapping[int(min_result.status)]}")
                else:
                    print(f"message: {min_result.message}")

                # See if BFGS did anything meaningful, if so override current estimate
                # if min_result.success: # and min_result.status not in [-1, 5]:
                if (
                    min_result.success
                    and min_result.status not in [-1]
                    and not jnp.isnan(min_result.x).any()
                ):
                    final_nll = min_result.fun
                    final_parameters = (
                        JointModel.parameter_type.from_flat_parameter_vector(
                            min_result.x
                        )
                    )
                else:
                    print("BFGS Optimization failed!")

        # Check if that joint type is better than any previous
        if final_nll < best_nll:
            best_nll = final_nll
            best_parameters = final_parameters
            best_joint_model = JointModel

    print(f"{best_joint_model = } with {best_parameters = }")
    base_transform = best_parameters.base_transform
    twist = best_parameters.to_twist()

    nll_, outlier_ratio = calculate_nll(
        best_parameters,
        stacked_datapoints,
        variance_pos=variance_pos,
        variance_ori=variance_ori,
        JointModel=best_joint_model,
        flat_parameters=False,
        return_outlier_ratio=True,
    )

    print(f"{outlier_ratio = }")
    if "outlier_ratio" in aux_data.keys():
        aux_data["outlier_ratio"] = outlier_ratio

    if "joint_states" in aux_data.keys():
        joint_states = jax.vmap(
            functools.partial(best_joint_model.backward, best_parameters)
        )(stacked_datapoints)
        aux_data["joint_states"] = joint_states

        # print("Testing whether the joint states stays constant")
        # forward_poses_sturm = jax.vmap(
        #     functools.partial(best_joint_model.forward, best_parameters)
        # )(joint_states)

        # def forward_twist(joint_state):
        #     return base_transform @ jaxlie.SE3.exp(joint_state * twist)

        # forward_poses_twist = jax.vmap(forward_twist)(joint_states)

        # residuals = jax.vmap(jaxlie.manifold.rminus)(
        #     forward_poses_sturm, forward_poses_twist
        # )
        # print(residuals)

    # Is this correct?
    return twist, base_transform, aux_data
