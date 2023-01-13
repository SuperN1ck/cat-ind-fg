import itertools
import random
from enum import Enum

import jax.numpy as jnp
import jax.random
import numpy as onp
import pandas
import jaxlie
from matplotlib.animation import FuncAnimation

from typing import Dict, Any, List, Union


class MotionType(Enum):
    RIGID = "rigid"
    TRANS = "translation"
    ROT = "rotation"
    HELIC = "helical"


numeric_mapping = {
    MotionType.RIGID: 0,
    MotionType.TRANS: 1,
    MotionType.ROT: 2,
    MotionType.HELIC: 3,
}


def get_motion_type_from_twist(
    twist: jnp.ndarray, threshold: float = 1e-3
) -> MotionType:
    norm_trans = jnp.linalg.norm(twist[:3])
    norm_rot = jnp.linalg.norm(twist[3:])
    scalar_rot_trans = twist[3:].T @ twist[:3]
    # print("<rot, trans>", scalar_rot_trans)

    if norm_rot < threshold and norm_trans > threshold:
        return MotionType.TRANS
    elif norm_rot > threshold:
        if jnp.abs(scalar_rot_trans) < threshold:
            return MotionType.ROT
        else:
            return MotionType.HELIC
    return MotionType.RIGID


def get_motion_type_from_decoupled_parameters(parameters, threshold=1e-1):
    assert parameters.shape == (8,)
    rotation_portion = parameters[6]
    translation_portion = parameters[7]

    if (
        onp.abs(rotation_portion) < threshold
        and onp.abs(translation_portion) > threshold
    ):
        return MotionType.TRANS
    elif (
        onp.abs(rotation_portion) > threshold
        and onp.abs(translation_portion) < threshold
    ):
        return MotionType.ROT
    elif (
        onp.abs(rotation_portion) < threshold
        and onp.abs(translation_portion) > threshold
    ):
        return MotionType.RIGID
    else:
        return MotionType.HELIC


def get_pitch(twist: jnp.ndarray):
    trans = twist[..., :3]
    rot = twist[..., 3:]
    norm_trans = jnp.linalg.norm(trans, axis=-1, keepdims=True)
    norm_rot = jnp.linalg.norm(rot, axis=-1, keepdims=True)
    return jnp.where(norm_rot < 1e-6, jnp.inf, norm_trans / norm_rot)


def general_dot(a, b, axis=-1):
    return jnp.sum(a * b, axis=axis, keepdims=True)


def get_normalized_scalar(twist: jnp.ndarray):
    trans = twist[..., :3]
    rot = twist[..., 3:]

    norm_trans = jnp.linalg.norm(trans, axis=-1, keepdims=True)
    norm_rot = jnp.linalg.norm(rot, axis=-1, keepdims=True)

    return jnp.where(
        jnp.logical_or(norm_rot < 1e-6, norm_trans < 1e-6),
        0.0,
        general_dot(trans / norm_trans, rot / norm_rot),
    )
    # return jnp.dot(trans / norm_trans, rot / norm_rot)
    # return jnp.einsum("ij,ij->i", trans / norm_trans, rot / norm_rot)


def get_norm_factor(twist: jnp.ndarray):
    trans = twist[:3]
    rot = twist[3:]
    norm_trans = jnp.linalg.norm(trans)
    norm_rot = jnp.linalg.norm(rot)
    if norm_rot < 1e-3:
        return norm_trans
    else:
        return norm_rot


def transform_twist_rel(
    twist: jnp.ndarray, relative_transform: jaxlie.SE3
) -> jnp.ndarray:
    """
    Given a twist in frame i this transform it to j
    j^T_i @ twist_i
    """
    return relative_transform.adjoint() @ twist


# def clean_twist(twist_: jnp.ndarray):
#     return normalize_twist(twist_, threshold=1e-1)


def normalize_twist(twist_: jnp.ndarray, threshold=1e-3):
    trans = twist_[:3]
    rot = twist_[3:]
    norm_trans = jnp.linalg.norm(trans)
    norm_rot = jnp.linalg.norm(rot)

    # if norm_rot > 1e-6:
    #     return twist_ / norm_rot
    # elif norm_trans > 1e-6:
    #     return twist_ / norm_trans
    # else:
    #     return twist_

    # print(f"{norm_trans = }")
    # print(f"{norm_rot = }")

    return jnp.where(
        norm_rot > threshold,
        twist_ / norm_rot,
        jnp.where(norm_trans > threshold, twist_ / norm_trans, twist_),
    )


def clean_twist(twist_: jnp.ndarray, **kwargs) -> jnp.ndarray:
    twist_threshold_pitch = kwargs.get("twist_threshold_pitch", 1e-1)
    twist_threshold_trans = kwargs.get("twist_threshold_trans", 1.5e-1)

    # Remove values close to 0
    twist_ = jnp.where(jnp.abs(twist_) < 1e-6, jnp.zeros_like(twist_), twist_)
    # "Meta" Info of Twist
    pitch = get_pitch(
        twist_
    )  # Determintes the ratio between translation/rotation movements
    scalar = get_normalized_scalar(
        twist_
    )  # When close to 0, rotation otherwise helical movement

    # print(f"{pitch = }")
    # print(f"{scalar = }")

    trans = twist_[:3]
    rot = twist_[3:]
    norm_trans = jnp.linalg.norm(trans)
    norm_rot = jnp.linalg.norm(rot)

    # print(f"{norm_trans = }")
    # print(f"{norm_rot = }")

    if norm_rot < twist_threshold_trans:
        if norm_trans < twist_threshold_trans:  # Rigid
            # print("Cleaned to Rigid")
            twist = jnp.zeros(6)
        else:  # Translation
            # print("Cleaned to Translation 1")
            twist = jnp.concatenate((trans / norm_trans, jnp.zeros(3)))
    else:
        # print(f"Pitch {pitch} {1 / pitch} < {threshold}")
        if 1 / pitch < (
            twist_threshold_pitch
        ):  # Special case where the center of rotation is very far away--> will be a translation motion along the tangent of the rotation axis
            # print("Cleaned to Translation 2")
            twist = jnp.concatenate((trans / norm_trans, jnp.zeros(3)))
        else:
            if (
                jnp.abs(scalar) < twist_threshold_trans
            ):  # Rotation --> The axis are orthogonal
                # We should ensure that dot projection becomes 0 --> "Reject the vector" (inverse of projection)
                new_trans = jnp.cross(
                    jnp.cross(rot, trans), rot
                )  # Double cross product removes the projective part
                norm_new_trans = jnp.linalg.norm(new_trans)
                scaling = norm_trans / norm_new_trans
                new_trans *= scaling

                # print(f"{trans = } --> {new_trans = }")

                twist = jnp.concatenate((new_trans, rot)) / norm_rot
            else:  # Helic
                # print("Cleaned to Helic")
                # print(f"twist = {twist_} / {norm_rot}")
                twist = twist_ / norm_rot
    return twist


def get_line_parameters(twist: jnp.ndarray):
    assert twist.shape[0] == 6

    motion_type = get_motion_type_from_twist(twist)
    if motion_type == MotionType.RIGID:
        n, p = jnp.zeros((3,)), jnp.zeros((3,))
    elif motion_type == MotionType.TRANS:
        n = twist[:3]
        p = jnp.zeros((3,))
    else:
        n = twist[3:]
        p = -jnp.cross(twist[:3], twist[3:])
    return n, p


def get_angle(v1: jnp.ndarray, v2: jnp.ndarray):
    """
    Assumes v1 and v2 are unit vectors
    """
    norm_v1 = jnp.linalg.norm(v1)
    norm_v2 = jnp.linalg.norm(v2)
    dot_ = jnp.dot(v1, v2) / (
        norm_v1 * norm_v2
    )  # For whatever reason this can become smaller than -1 or bigger than 1..
    angle_ = jnp.arccos(jnp.clip(dot_, -1.0, 1.0))
    # Account for flipped axis
    angle = (angle_ - jnp.pi) if angle_ > jnp.pi / 2 else angle_
    return angle


def get_distance(p1, l1, p2, l2):
    """
    ScrewNet Jain et al. 2021, Implementation
    Assumes l1 and l2 are unit vectors
    Lines are given by: line1 = p1 + \lambda_1 l1
    And line2 = p2 + \lambda_2 l2
    """
    # l1 and l2 intersect --> covered by last case
    m1 = jnp.cross(l1, p1)
    m2 = jnp.cross(l2, p2)
    if jnp.dot(l1, l2) < 0:
        l2 = -l2
        m2 = -m2

    if jnp.abs(jnp.abs(jnp.dot(l1, l2)) - 1.0) < 1e-6:
        return jnp.linalg.norm(jnp.cross(l1, (m1 - m2)))
    else:
        return jnp.linalg.norm(jnp.dot(l1, m2) + jnp.dot(l2, m1)) / jnp.linalg.norm(
            jnp.cross(l1, l2)
        )


def compute_twist_center(twist: Union[jnp.ndarray, onp.ndarray]) -> onp.ndarray:
    """Finds a translation p such that when applied to the twist, the twist's
    linear component is 0."""
    if isinstance(twist, jnp.ndarray):
        twist = onp.array(twist)
    v = twist[:3]
    w = twist[3:]

    p = onp.cross(w, v) / w.dot(w)

    # print("p", p)
    # print("p x w", onp.cross(p, w))
    # print("v - p x w", v - onp.cross(p, w))

    # Verify that the translated twist's linear component is 0.
    assert (
        onp.linalg.norm(onp.cross(w, v - onp.cross(p, w))) < 1e-5
    ), f"{onp.linalg.norm(onp.cross(w, v - onp.cross(p, w)))}"

    return p


def normalize_linear_motion(twist: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
    """Normalizes the twist so that when it is translated to the given point,
    resulting linear motion has unit norm."""
    v = twist[:3]
    w = twist[3:]

    v_w = jnp.cross(w, point)
    norm = jnp.linalg.norm(v + v_w)

    return twist / norm


def skew(omega: jnp.ndarray) -> jnp.ndarray:
    """Returns the skew-symmetric form of a length-3 vector."""
    wx, wy, wz = omega
    return jnp.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def mean_pose(poses: List[jaxlie.SE3]) -> jaxlie.SE3:
    """Computes the average pose."""
    quaternions = jnp.array([pose.wxyz_xyz[:4] for pose in poses])
    positions = jnp.array([pose.wxyz_xyz[4:] for pose in poses])

    # Average quaternion is the eigenvector of Q^T Q corresponding to the
    # largest eigenvalue.
    QTQ = quaternions.transpose() @ quaternions
    _, eigvecs = jnp.linalg.eigh(QTQ)
    wxyz = eigvecs[:, -1]
    quaternion = jaxlie.SO3(wxyz / jnp.linalg.norm(wxyz))

    position = positions.mean(axis=0)

    return jaxlie.SE3.from_rotation_and_translation(quaternion, position)


class RandomKeyGenerator:
    key: jnp.ndarray

    def __init__(self, seed: int = 0):
        self.key = self.single_key(seed)

    def next_key(self):
        self.key, to_use = jax.random.split(self.key)
        return to_use

    @staticmethod
    def single_key(seed: int = 0):
        if seed == -1:
            seed = random.getrandbits(32)
        return jax.random.PRNGKey(seed=seed)


def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None


@jax.jit
def batch_samples(samples):
    return jax.tree_map(lambda *leaves: jnp.stack(leaves, axis=0), *samples)


@jax.jit
def get_sample(batch, i):
    return jax.tree_map(lambda leaf: leaf[i, ...], batch)


# @jax.jit
def weighted_mean_and_var(values, weights, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    https://stackoverflow.com/a/2415343
    """
    average = jnp.average(values, weights=weights, axis=axis)
    # Fast and numerically precise:
    variance = jnp.average((values - average) ** 2, weights=weights, axis=axis)
    return average, variance


def inc_dict(dict_, key):
    dict_[key] = dict_.get(key, 0) + 1
    return dict_


class KeyDict(dict):
    @staticmethod
    def __missing__(key):
        key


def extract_dict(in_dict, key_dict: KeyDict):
    return {new_key: in_dict[old_key] for old_key, new_key in key_dict.items()}


def dict_product(dict_in):
    keys = dict_in.keys()
    vals = dict_in.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def dict_subset(
    dict_in: Dict[Any, Any], subset_keys: Union[List[Any], Any]
) -> Dict[Any, Any]:
    if not isinstance(subset_keys, List):
        subset_keys = [subset_keys]
    return {key: dict_in[key] for key in subset_keys if key in dict_in.keys()}


def stack_transformations(trajectory):
    return [jaxlie.SE3.exp(cluster.transformation) for cluster in trajectory]


def stack_centers(trajectory):
    return [cluster.center for cluster in trajectory] + [
        trajectory[-1].center_forwarded
    ]


def create_rotation(fig, path, elevs, azims, dpi=80, duration_ms=5000):
    assert elevs.shape == azims.shape
    ax = fig.gca()

    def update(i):
        ax.view_init(elevs[i], azims[i])

    anim = FuncAnimation(
        fig,
        update,
        frames=onp.arange(0, elevs.shape[0]),
        interval=duration_ms / elevs.shape[0],
    )
    anim.save(path, dpi=dpi, writer="imagemagick")


def bgr_to_rgb(img):
    assert img.shape[-1] == 3
    return img[..., [2, 1, 0]]


def rgb_to_bgr(img):
    assert img.shape[-1] == 3
    return img[..., [2, 1, 0]]


def set_random_seed(seed: int) -> None:
    """Set seeds for `random` and `numpy`."""
    random.seed(seed)
    onp.random.seed(seed)
