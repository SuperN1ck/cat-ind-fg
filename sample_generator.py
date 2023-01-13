import os
import pickle
import random
from typing import Dict, List, NamedTuple, Optional, Tuple

import igraph
import jax
import jax.numpy as jnp
import jax.random
import jax_dataclasses
import numpy as onp
from jaxlie import SE3, SO3, manifold

from helpers import (
    MotionType,
    RandomKeyGenerator,
    normalize_twist,
    transform_twist_rel,
)


class JointConnection(NamedTuple):
    from_id: str  # SE3Observations ID
    to_id: str  # SE3Observations ID
    via_id: Optional[
        str
    ] = None  # Can be used to define a SE3Observations transformation ID


def apply_noise(T: SE3, key: jnp.ndarray, variance: float = 10.0) -> SE3:
    dim = 6
    noise = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros((dim,)), cov=(variance + 1e-6) * jnp.eye(dim)
    )
    return manifold.rplus(T, noise)


def get_random_twist(motion_type: MotionType, key: jnp.ndarray):
    key_0, key_1 = jax.random.split(key)
    twist_: jnp.ndarray
    if motion_type == MotionType.RIGID:
        twist_ = jnp.zeros(6)
    elif motion_type == MotionType.TRANS:
        twist_ = jnp.concatenate(
            (
                jax.random.uniform(key=key_0, shape=(3,), minval=-1.0, maxval=1.0),
                jnp.zeros(3),
            )
        )
    elif motion_type == MotionType.ROT:
        trans_part = jax.random.uniform(key=key_0, shape=(3,), minval=-1.0, maxval=1.0)
        rot_part = jnp.cross(
            trans_part,
            jax.random.uniform(key=key_1, shape=(3,), minval=-1.0, maxval=1.0),
        )
        twist_ = jnp.concatenate((trans_part, rot_part))
    else:  # motion_type == MotionType.HELICAL
        twist_ = jax.random.uniform(key=key_0, shape=(6,), minval=-1.0, maxval=1.0)
    return normalize_twist(twist_)


def sample_se3(key, position_range: float) -> SE3:
    key0, key1 = jax.random.split(key)
    return SE3.from_rotation_and_translation(
        rotation=SO3.sample_uniform(key0),
        translation=jax.random.uniform(
            key=key1, shape=(3,), minval=-position_range, maxval=position_range
        ),
    )


def get_canonical_twist(motion_type: MotionType, key: jnp.ndarray) -> jnp.ndarray:
    if motion_type == MotionType.RIGID:
        twist = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif motion_type == MotionType.TRANS:
        twist = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif motion_type == MotionType.ROT:
        twist = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    elif motion_type == MotionType.HELIC:
        pitch = jnp.random.uniform(key=key, shape=(2,))
        pitch /= pitch.max()
        twist = jnp.array([0.0, 0.0, pitch[0], 0.0, 0.0, pitch[1]])
    else:
        raise NotImplementedError()

    return twist


def plot_samples(Ts_world_first_obs: List[SE3], Ts_world_second_obs: List[SE3]) -> None:
    import matplotlib.pyplot as plt

    def set_axes_equal(ax: plt.Axes) -> None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = onp.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = onp.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = onp.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_frames(
        ax: plt.Axes, Ts: List[SE3], label: str, axis_size: float = 0.1
    ) -> None:
        points = onp.array([T.translation() for T in Ts])
        axes = axis_size * onp.stack([T.rotation().as_matrix().T for T in Ts], axis=1)

        ax.scatter(*points.T, label=label)
        ax.quiver(*points.T, *axes[..., 0], color="r")
        ax.quiver(*points.T, *axes[..., 1], color="g")
        ax.quiver(*points.T, *axes[..., 2], color="b")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(projection="3d")
    plot_frames(ax, Ts_world_first_obs, "first")
    plot_frames(ax, Ts_world_second_obs, "second")
    set_axes_equal(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    plt.show()


@jax_dataclasses.pytree_dataclass
class Sample:
    # TODO refacotor such that they are all attributes of a graph --> more complex structure?
    observations: Dict[str, List[SE3]]
    gts: Dict[str, List[SE3]]
    twists: Dict[str, jnp.ndarray]
    base_transforms: Dict[str, SE3]
    structure: Dict[str, JointConnection]

    @staticmethod
    def generate_random(
        observation_amount: int = 100,
        sample_length: float = 0.5,
        stddev_pos: float = 0.005,
        stddev_ori: float = 0.02,
        N: int = 2,  # Amount of parts
        motion_type: MotionType = MotionType.TRANS,
        seed: int = None,
        old_method: bool = False,
        variance_old: float = 1e-1,
    ):
        if not seed:
            seed = random.getrandbits(32)
        print("New sample with seed {}".format(seed))
        rng_key_gen = RandomKeyGenerator(seed=seed)

        if old_method:
            # full_graph = igraph.Graph.Full(n=N, directed=False, loops=False)
            # random_weights = onp.random.uniform(size=N)
            # structure = full_graph.spanning_tree(weights=random_weights)

            # Get base transformation from first body (base) to second (part, that gets actuated)
            T_world_first = SE3.sample_uniform(rng_key_gen.next_key())
            T_first_world = T_world_first.inverse()
            # print("T_world_first\n", T_world_first.as_matrix())
            # print("T_first_world\n", T_first_world.as_matrix())
            T_world_second = SE3.sample_uniform(rng_key_gen.next_key())
            T_first_second_zero = T_first_world @ T_world_second

            Ts_first_second = []
            Ts_world_first = []
            Ts_world_second = []
            joint_states = []

            twist = get_random_twist(motion_type, rng_key_gen.next_key())

            for joint_state in onp.linspace(0, sample_length, observation_amount):
                twist_i = twist * joint_state
                T_twist = SE3.exp(twist_i)
                T_first_second = T_first_second_zero @ T_twist

                T_world_second = T_world_first @ T_first_second

                Ts_world_first.append(T_world_first)
                Ts_world_second.append(T_world_second)
                Ts_first_second.append(T_first_second)
                joint_states.append(joint_state)

            # Add noise to trajectories, can this be parallelized?
            Ts_world_first_obs = [
                apply_noise(T, rng_key_gen.next_key(), variance=variance_old)
                for T in Ts_world_first
            ]
            Ts_world_second_obs = [
                apply_noise(T, rng_key_gen.next_key(), variance=variance_old)
                for T in Ts_world_second
            ]
            Ts_first_second_obs = [
                apply_noise(T, rng_key_gen.next_key(), variance=variance_old)
                for T in Ts_first_second
            ]
        else:
            # Randomly sample T_joint_second instead of T_world_second.
            T_first_joint = sample_se3(rng_key_gen.next_key(), position_range=0.5)
            T_joint_second = sample_se3(rng_key_gen.next_key(), position_range=0.5)
            T_first_second_zero = T_first_joint

            # Joint articulation transforms.
            qs = onp.linspace(0, sample_length, observation_amount)
            twist = get_canonical_twist(motion_type, rng_key_gen.next_key())
            Ts_joint = [SE3.exp(q * twist) for q in qs]

            # Center all points.
            Ts_first_second = [
                T_first_joint @ T_joint @ T_joint_second for T_joint in Ts_joint
            ]
            centers = [T.translation() for T in Ts_first_second]
            center = onp.mean(centers, axis=0)

            # Transform entire articulated body. Set the translation such that
            # the mean position for all observations (of both bodies) is 0.
            R_world_first = SO3.sample_uniform(rng_key_gen.next_key())
            T_world_first = SE3.from_rotation_and_translation(
                R_world_first, R_world_first.as_matrix() @ -center / 2
            )

            # Observation noise.
            twist_std = onp.array([stddev_pos] * 3 + [stddev_ori] * 3)
            twist_variance = onp.diag(twist_std * twist_std)
            noise_twists = jax.random.multivariate_normal(
                key=rng_key_gen.next_key(),
                mean=jnp.zeros_like(twist_std),
                cov=twist_variance,
                shape=(2 * observation_amount,),
            )
            Ts_noise = [SE3.exp(twist_i) for twist_i in noise_twists]
            Ts_first_obs = Ts_noise[:observation_amount]
            Ts_second_obs = Ts_noise[observation_amount:]

            # Combine transformations.
            Ts_world_first = [T_world_first] * observation_amount
            Ts_world_second = [
                T_world_first @ T_first_second for T_first_second in Ts_first_second
            ]
            Ts_world_first_obs = [
                T_world_first @ T_first_obs
                for (T_world_first, T_first_obs) in zip(Ts_world_first, Ts_first_obs)
            ]
            Ts_world_second_obs = [
                T_world_second @ T_second_obs
                for (T_world_second, T_second_obs) in zip(
                    Ts_world_second, Ts_second_obs
                )
            ]
            Ts_first_second_obs = [
                T_world_first.inverse() @ T_world_second
                for (T_world_first, T_world_second) in zip(
                    Ts_world_first_obs, Ts_world_second_obs
                )
            ]

        # plot_samples(Ts_world_first_obs, Ts_world_second_obs)

        observations = {
            "first": Ts_world_first_obs,
            "second": Ts_world_second_obs,
            "first_second": Ts_first_second_obs,
        }

        gts = {
            "first": Ts_world_first,
            "second": Ts_world_second,
            "first_second": Ts_first_second,
        }

        twists = {"first_second": twist}
        base_transforms = {"first_second": T_first_second_zero}
        structure = {
            "first_second": JointConnection(
                from_id="first", to_id="second", via_id="first_second"
            )
        }

        return Sample(
            observations=observations,
            gts=gts,
            twists=twists,
            base_transforms=base_transforms,
            structure=structure,
        )
