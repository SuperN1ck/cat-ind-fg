from typing import Callable, Tuple, Dict, List

import jax.numpy as jnp
import jaxlie
import numpy as onp
import jax

import helpers


def joint_error(
    gt_base_transform: jaxlie.SE3,
    gt_twist: jnp.ndarray,
    est_base_transform: jaxlie.SE3,
    est_twist: jnp.ndarray,
    verbose=False,
    **kwargs,
) -> Tuple[bool, jnp.ndarray, Dict[str, helpers.MotionType]]:
    """
    Returns a tuple containing
    (Correct Joint Type, [Angle Error, Distance Error])
    Careful! This neglects the case that a rotation motion with a very far center is equivalent to a translation motion
    """

    # First check for correct joint type
    # Transform
    gt_twist_common = helpers.normalize_twist(
        helpers.transform_twist_rel(gt_twist, gt_base_transform)
    )
    est_twist_common = helpers.clean_twist(
        helpers.normalize_twist(
            helpers.transform_twist_rel(est_twist, est_base_transform)
        ),
        **kwargs,
    )

    gt_motion_type = helpers.get_motion_type_from_twist(gt_twist_common)
    est_motion_type = helpers.get_motion_type_from_twist(est_twist_common)

    motion_types = {"gt": gt_motion_type, "est": est_motion_type}
    correct_motion_type = gt_motion_type == est_motion_type

    if verbose:
        print(f"{gt_motion_type = }\n\t{gt_twist_common = }")
        print(f"{est_motion_type = }\n\t{est_twist_common = }")

    n_gt, p_gt = helpers.get_line_parameters(gt_twist_common)
    n_est, p_est = helpers.get_line_parameters(est_twist_common)

    if gt_motion_type == helpers.MotionType.RIGID:
        return correct_motion_type, jnp.zeros((2,)), motion_types
    elif gt_motion_type == helpers.MotionType.TRANS:
        # Careful! Does not handle the case of distant rotation?
        return (
            correct_motion_type,
            jnp.array([helpers.get_angle(n_gt, n_est), 0.0]),
            motion_types,
        )
    else:  # Rot / Helical
        angle = helpers.get_angle(n_gt, n_est)
        distance = helpers.get_distance(p_gt, n_gt, p_est, n_est)
        return (
            correct_motion_type,
            jnp.array([angle, distance]),
            motion_types,
        )

    return True, jnp.array([jnp.inf, jnp.inf]), motion_types


def translation_direction_metric(
    gt_base_transform: jaxlie.SE3,
    gt_twist: jnp.ndarray,
    est_base_transform: jaxlie.SE3,
    est_twist: jnp.ndarray,
    poses: List[jaxlie.SE3],
):

    gt_twist_common = helpers.transform_twist_rel(gt_twist, gt_base_transform)
    est_twist_common = helpers.transform_twist_rel(est_twist, est_base_transform)

    def get_angle(pose=jaxlie.SE3.identity):
        common_frame = jaxlie.SE3.from_rotation_and_translation(
            rotation=jaxlie.SO3.identity(),
            translation=-pose.translation(),
        )

        gt_direction = helpers.transform_twist_rel(
            gt_twist_common,
            common_frame,
        )[:3]
        est_direction = helpers.transform_twist_rel(
            est_twist_common,
            common_frame,
        )[:3]
        return helpers.get_angle(gt_direction, est_direction)

    direction_angles = onp.array([get_angle(pose) for pose in poses])

    # TODO Investigate to make things faster?
    # try:
    #     poses_stacked = helpers.batch_samples(poses)
    #     direction_angles_ = jax.vmap(get_angle)(
    #         gt_joint_states, est_joint_states, poses_stacked
    #     )
    # except:
    #     print("Failed jax.vmap")

    # print(f"{direction_angles = }")

    return onp.nanmean(
        onp.abs(direction_angles)
    )  # nanmean because twist can become zero!


def _get_grasp_path_params(
    gt_twist: jnp.ndarray, observed_points: onp.ndarray
) -> Tuple[onp.ndarray, float]:
    """Infers the grasp path params (x_0, q_max) from the observed grasp poses.
    Ideally, these parameters should be given from the ground truth data
    generation."""
    w = onp.array(gt_twist[3:])
    w_norm = onp.linalg.norm(w)
    if w_norm < 0.01:
        v_n = onp.array(gt_twist[:3])
        v_n /= onp.linalg.norm(v_n)
        zs = observed_points.dot(v_n)

        idx_q_min = onp.argmin(zs)
        idx_q_max = onp.argmax(zs)
        q_max = zs[idx_q_max] - zs[idx_q_min]

        xs_r = observed_points - zs[:, None] * v_n[None, :]
        x_r = xs_r.mean(axis=0)

        x_0_z = zs[idx_q_min] * v_n
        x_0 = x_0_z + x_r

        return x_0, q_max

    # Compute radius around rotation axis.
    w_n = w / w_norm
    rotation_center = helpers.compute_twist_center(gt_twist)
    rs = observed_points - rotation_center[None, :]
    rs -= rs.dot(w_n)[:, None] * w_n[None, :]
    r = onp.linalg.norm(rs, axis=1).mean()

    # Project rs to 2D plane.
    _, _, vt = onp.linalg.svd(rs)
    if onp.linalg.det(vt) < 0:
        vt[0] *= -1
    if vt[2].dot(w_n) < 0:  # Make sure z-axis is pointing in rotation axis.
        vt[1:] *= -1  # Flip 180 about x axis.
    assert onp.linalg.det(vt) > 0
    v = vt[:2].T

    def rotate_2d(xs: onp.ndarray) -> onp.ndarray:
        return xs @ v

    # Find range of joint angles.
    rs_2d = rotate_2d(rs)
    thetas = onp.arctan2(rs_2d[:, 1], rs_2d[:, 0])  # Range [-pi, pi].

    # Find the index of the point whose distance to the nearest preceding point
    # about the rotation axis is the largest. This point marks the lower bound of
    # the joint range.
    dthetas = onp.zeros_like(thetas)
    idx_thetas = onp.argsort(thetas)
    sorted_thetas = onp.concatenate(
        ([thetas[idx_thetas[-1]] - 2 * onp.pi], thetas[idx_thetas])
    )
    dthetas[idx_thetas] = sorted_thetas[1:] - sorted_thetas[:-1]
    assert (dthetas >= 0).all()
    idx_q_min = onp.argmax(dthetas)
    idx_q_max = idx_thetas[onp.where(idx_thetas == idx_q_min)[0][0] - 1]

    # Compute the starting point with distance r from the rotation axis.
    x_0 = observed_points[idx_q_min] - rotation_center
    x_0_w = x_0.dot(w_n) * w_n
    x_0_r = x_0 - x_0_w
    x_0_r *= r / onp.linalg.norm(x_0_r)
    x_0 = x_0_r + x_0_w + rotation_center

    # Compute the upper bound of the joint angle.
    q_max = (thetas[idx_q_max] - thetas[idx_q_min]) % (2 * onp.pi)
    assert q_max >= 0, f"{q_max} >= 0"

    return x_0, q_max


def _get_grasp_path_fn(
    gt_twist: jnp.ndarray, x_0: onp.ndarray
) -> Callable[[onp.ndarray], jnp.ndarray]:
    """Creates the function x(q) = exp(q v) x_0."""
    # Augment point.
    x_0 = onp.concatenate((x_0, [1]))

    def grasp_path_fn(qs: onp.ndarray) -> jnp.ndarray:
        """Computes x(q) = exp(q v) x_0."""
        Ts = jnp.array([jaxlie.SE3.exp(q * gt_twist).as_matrix() for q in qs])
        xs = Ts @ x_0

        return xs[:, :3]

    return grasp_path_fn


def _generate_grasp_points(
    gt_twist: jnp.ndarray, observed_points: onp.ndarray, num_samples: int
) -> jnp.ndarray:
    """Generates `num_samples` equally spaced points along the trajectory
    spanned by `observed_points`."""
    x_0, q_max = _get_grasp_path_params(gt_twist, observed_points)
    grasp_path = _get_grasp_path_fn(gt_twist, x_0)

    qs = onp.linspace(0, q_max, num_samples)
    grasp_points = grasp_path(qs)

    return grasp_points


def _compute_linear_motion(twist: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    """Computes the unit norm linear motion Ad(twist)_v at x."""
    v = twist[:3][None, :]
    w_x = helpers.skew(twist[3:])

    linear_motion = v - xs @ w_x
    linear_motion /= jnp.linalg.norm(linear_motion, axis=-1)[..., None]
    return linear_motion


def _compute_linear_motion_similarity(
    gt_twist: jnp.ndarray, pred_twist: jnp.ndarray, grasp_points: jnp.ndarray
) -> jnp.ndarray:
    """Computes the dot product between the ground truth and predicted
    linear motions at x."""
    gt_linear_motion = _compute_linear_motion(gt_twist, grasp_points)
    pred_linear_motion = _compute_linear_motion(pred_twist, grasp_points)

    return (gt_linear_motion * pred_linear_motion).sum(axis=-1)


def twist_metrics(
    gt_base_transform: jaxlie.SE3,
    gt_twist: jnp.ndarray,
    pred_base_transform: jaxlie.SE3,
    pred_twist: jnp.ndarray,
    poses: List[jaxlie.SE3],
    num_samples: int = 100,
) -> Tuple[float, float]:
    # Sanity check: pred_twist should be not rigid
    if helpers.get_motion_type_from_twist(pred_twist) == helpers.MotionType.RIGID:
        print("Encountered rigid twist")
        return 0, onp.pi / 2

    """Computes the average linear motion similarity and angle error along the
    path traced out by the given ground truth poses."""
    # Format input data.
    gt_twist = helpers.transform_twist_rel(gt_twist, gt_base_transform)
    pred_twist = helpers.transform_twist_rel(pred_twist, pred_base_transform)
    observed_points = onp.array([grasp_pose.translation() for grasp_pose in poses])
    grasp_points = _generate_grasp_points(gt_twist, observed_points, num_samples)

    # Normalize twists so their linear components are well-conditioned along the grasp path.
    grasp_midpoint = grasp_points[len(grasp_points) // 2]
    gt_twist = helpers.normalize_linear_motion(gt_twist, grasp_midpoint)
    pred_twist = helpers.normalize_linear_motion(pred_twist, grasp_midpoint)

    # Compute the linear motion dot products along the grasp path.
    similarities = _compute_linear_motion_similarity(gt_twist, pred_twist, grasp_points)
    if (similarities < 0).sum() >= len(similarities) / 2:
        similarities *= -1

    # Compute the angle errors along the grasp path.
    angle_errors = jnp.arccos(jnp.clip(similarities, -1, 1))

    # Integrate along the grasp path.
    similarity = jnp.clip(similarities.mean(), -1, 1)
    angle_error = angle_errors.mean()

    # assert similarity >= 0
    # assert angle_error >= 0
    if similarity < 0:
        print(f"{similarity = } < 0 encoutered!")
    if angle_error < 0:
        print(f"{angle_error = } < 0 encoutered!")

    return similarity, angle_error
