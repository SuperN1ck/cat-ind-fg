import abc
from itertools import product
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxfg
import jaxlie
import numpy as onp
import helpers
from jax.scipy.optimize import minimize
from jaxlie import SE3, SO3
from overrides import EnforceOverrides, overrides

import os

print(os.path.dirname(helpers.__file__))


@jax_dataclasses.pytree_dataclass
class BaseJointParameters:
    base_transform: SE3

    @abc.abstractclassmethod
    def get_flat_parameter_vector(self):
        pass

    @abc.abstractstaticmethod
    def from_flat_parameter_vector(flat_vector):
        pass

    @staticmethod
    def _flatten(v):
        as_dict = vars(v)
        return tuple(as_dict.values()), tuple(as_dict.keys())

    @abc.abstractstaticmethod
    def _unflatten(treedef, children):
        return BaseJointParameters(**dict(zip(treedef, children)))  # type: ignore

    @abc.abstractclassmethod
    def to_twist(self) -> jnp.ndarray:
        pass


class BaseJoint(abc.ABC, EnforceOverrides):
    min_samples: int
    motion_type: helpers.MotionType

    @abc.abstractstaticmethod
    def estimate_parameters(
        samples: List[SE3], visualize: Optional[bool] = False
    ) -> BaseJointParameters:
        pass

    @abc.abstractstaticmethod
    def forward(parameters: BaseJointParameters, joint_state: float):
        pass

    @abc.abstractstaticmethod
    def backward(parameters: BaseJointParameters, transformation: SE3):
        pass


#########
# RIGID #
#########
@jax_dataclasses.pytree_dataclass
class RigidJointParameters(BaseJointParameters):
    def get_flat_parameter_vector(self):
        return self.base_transform.log()

    @staticmethod
    def from_flat_parameter_vector(flat_vector):
        return RigidJointParameters(base_transform=SE3.exp(flat_vector))

    @overrides
    def to_twist(self) -> jnp.ndarray:
        return jnp.zeros((6,))


class RigidJoint(BaseJoint):
    min_samples = 1
    parameter_type = RigidJointParameters
    motion_type = helpers.MotionType.RIGID

    @overrides
    def estimate_parameters(samples, visualize=False) -> RigidJointParameters:
        return RigidJointParameters(base_transform=samples[0])

    @jax.jit
    @overrides
    def forward(parameters, joint_state):
        return parameters.base_transform

    @jax.jit
    @overrides
    def backward(parameters, transformation):
        # raise NotImplementedError
        return 0


#############
# PRISMATIC #
#############
@jax_dataclasses.pytree_dataclass
class PrismaticJointParameters(BaseJointParameters):
    # Additional axis
    axis: jnp.ndarray

    def get_flat_parameter_vector(self):
        return jnp.concatenate((self.base_transform.log(), self.axis))

    @staticmethod
    def from_flat_parameter_vector(flat_vector):
        return PrismaticJointParameters(
            base_transform=SE3.exp(flat_vector[:6]), axis=flat_vector[6:]
        )

    @overrides
    def to_twist(self) -> jnp.ndarray:
        return jnp.concatenate((self.axis, jnp.zeros((3,))))


class PrismaticJoint(BaseJoint):
    min_samples = 2
    parameter_type = PrismaticJointParameters
    motion_type = helpers.MotionType.TRANS

    @overrides
    def estimate_parameters(
        samples: List[SE3], visualize=False
    ) -> PrismaticJointParameters:
        base_transform = samples[0]
        points_world = jnp.array(
            [
                (base_transform.inverse() @ datapoint).translation()
                for datapoint in samples
            ]
        )
        center_points_world = jnp.mean(points_world, axis=0)

        _, dd, vv = jnp.linalg.svd(points_world - center_points_world)
        # Follow notation that columns represent our unit vectors for the frame
        vv = vv.T

        # print("dd {}\n".format(dd.shape), dd)
        # print("estimated translation axis", vv[:, 0])
        return PrismaticJointParameters(base_transform=base_transform, axis=vv[:, 0])

    @jax.jit
    @overrides
    def forward(parameters, joint_state: float):
        return parameters.base_transform @ SE3.from_rotation_and_translation(
            rotation=SO3.identity(), translation=parameters.axis * joint_state
        )

    @jax.jit
    @overrides
    def backward(parameters, transformation: SE3) -> float:
        return jnp.dot(
            (parameters.base_transform.inverse() @ transformation).translation(),
            # parameters.base_transform.translation(),
            parameters.axis,
        )


############
# REVOLUTE #
############
@jax_dataclasses.pytree_dataclass
class RevoluteJointParameters(BaseJointParameters):
    # Additional axis
    center_of_rotation: jnp.ndarray
    axis_of_rotation: jnp.ndarray

    def get_flat_parameter_vector(self):
        return jnp.concatenate(
            (self.base_transform.log(), self.center_of_rotation, self.axis_of_rotation)
        )

    @staticmethod
    def from_flat_parameter_vector(flat_vector):
        return RevoluteJointParameters(
            base_transform=SE3.exp(flat_vector[:6]),
            center_of_rotation=flat_vector[6:9],
            axis_of_rotation=flat_vector[9:],
        )

    @overrides
    def to_twist(self) -> jnp.ndarray:
        # Calculate the displacement of the rotation
        trans = jnp.cross(self.center_of_rotation, self.axis_of_rotation)
        return jnp.concatenate((trans, self.axis_of_rotation))


class RevoluteJoint(BaseJoint):
    min_samples = 3
    parameter_type = RevoluteJointParameters
    motion_type = helpers.MotionType.ROT

    @overrides
    def estimate_parameters(
        samples: List[SE3],
        visualize=False,
        method_rot="svd",  # ["twist", "svd"]
        method_center="direct",  # ["circle_optim", "direct"]
    ) -> RevoluteJointParameters:
        ones = jnp.ones((3,))

        base_transform = samples[0]
        points_world = jnp.array(
            [(base_transform.inverse() @ datapoint) @ ones for datapoint in samples]
        )
        # points_world = jnp.array(
        #     [
        #         (base_transform.inverse() @ datapoint).translation()
        #         for datapoint in samples
        #     ]
        # )
        # Old version?
        # points_world = jnp.array([datapoint @ ones for datapoint in samples])
        center_points_world = jnp.mean(points_world, axis=0)

        if method_rot == "twist":
            rot = (base_transform.inverse() @ samples[1]).rotation().log()
            vv_1 = jnp.cross(rot, jnp.array([rot[0], rot[1], rot[2] + 1]))
            vv_2 = jnp.cross(rot, vv_1)
            vv_ = jnp.vstack((vv_1, vv_2, rot)).T
            # Make orthonormal
            vv = vv_ / jnp.linalg.norm(
                vv_, axis=0
            )  # Normalize per row, but doesn't matter
        if (
            method_rot == "svd" or jnp.linalg.norm(rot) < 1e-6
        ):  # If there is no rotation the above `rot` will be just jnp.zeros(3), causing problems latter
            # Degenerates when all points are projected to a common point
            _, dd, vv = jnp.linalg.svd(points_world - center_points_world)
            vv = vv.T
            # print("dd", dd)
            # print("vv", vv)
            # Rotation axis is the axis with the smallest singular value, ideally dd[2] == 0!
            # Ensure that vv is a right-hand frame
            rot = jnp.cross(vv[:, 0], vv[:, 1])
            vv = jax.ops.index_update(vv, jax.ops.index[:, 2], rot)
            # print(vv)

        # Create a transformation that transforms our points to the svd frame
        T_world_svd = SE3.from_rotation_and_translation(
            translation=center_points_world, rotation=SO3.from_matrix(vv)
        )
        T_svd_world = T_world_svd.inverse()

        # Thinking transformations:
        # points_world = T_world_svd @ points_svd
        # points_svd = T_svd_world @ points_world
        # All points shoudl now be in a plane with normal rot
        points_svd = jax.vmap(T_svd_world.apply)(points_world)
        # Alternatively
        # points_svd = jax.vmap(SO3.from_matrix(vv.T).apply)(points_world - center_points_world)
        # points_svd = (vv.T @ (points_world - center_points_world).T).T # Need to apply inverse vv.T!

        if method_center == "circle_optim":
            # Fit a circle to determine the center of rotation --> Minimize
            def circle(x, points):
                return jnp.sum(
                    (
                        (points[:, 0] - x[0]) ** 2
                        + (points[:, 1] - x[1]) ** 2
                        - x[2] ** 2
                    )
                    ** 2
                )

            x0 = jnp.array([0.0, 0.0, jnp.max(jnp.abs(points_svd))])
            print("Starting minimizing...")
            min_result = minimize(
                circle, x0, args=(points_svd,), method="BFGS", options={"maxiter": 10}
            )
            print("min_result.fun", min_result.fun)
            print("min_result.x", min_result.x)
            print("min_result.nit", min_result.nit, "(Optimization iterations)")
            # The center of the circle is in our plane, therefore the z-component is zero
            center_circle_svd = jnp.concatenate((min_result.x[:2], jnp.zeros((1,))))

            # We transform our center back to the world frame
            center_circle_world = T_world_svd.apply(center_circle_svd)
            # Alternatively
            # center_circle_world = center_points_world + vv @ center_circle_svd
        elif method_center == "direct":
            frame = "svd"
            # frame = "world"

            def get_line_vectors(point_1, point_2):
                diff_ = point_2 - point_1
                on_point = point_1 + diff_ / 2
                diff = diff_ / jnp.linalg.norm(diff_)
                if frame == "svd":
                    return on_point, jnp.array([diff[1], -diff[0]])
                elif frame == "world":
                    return on_point, jnp.cross(rot, diff)

            if frame == "svd":
                points_svd_2d = points_svd[:, :2]
                on_point_0, normal_0 = get_line_vectors(
                    points_svd_2d[0], points_svd_2d[1]
                )
                on_point_1, normal_1 = get_line_vectors(
                    points_svd_2d[0], points_svd_2d[2]
                )
            elif frame == "world":
                on_point_0, normal_0 = get_line_vectors(
                    points_world[0], points_world[1]
                )
                on_point_1, normal_1 = get_line_vectors(
                    points_world[0], points_world[2]
                )

            # Check if lines are parallel or centers are close --> degeneration
            direct_degeneration = (
                jnp.abs(jnp.abs(jnp.dot(normal_0, normal_1)) - 1.0) < 1e-6
                or jnp.linalg.norm(on_point_0 - on_point_1) < 1e-6
            )

            if (
                direct_degeneration
            ):  # In the degenerated case we assign a "random" point as center
                center_circle_ = on_point_0
            else:
                # We have a skewed line setup and will find the two points corresponding to the closest distance between them.
                # The center of rotation is the midpoint of this connection line
                # See here: https://math.stackexchange.com/a/2213256

                e = on_point_0 - on_point_1
                A = (jnp.dot(normal_0, normal_1) ** 2) - (
                    jnp.dot(normal_0, normal_0) * jnp.dot(normal_1, normal_1)
                )
                param_0 = (
                    jnp.dot(normal_1, normal_1) * jnp.dot(normal_0, e)
                    - jnp.dot(normal_1, e) * jnp.dot(normal_0, normal_1)
                ) / A
                param_1 = (
                    -jnp.dot(normal_0, normal_0) * jnp.dot(normal_1, e)
                    + jnp.dot(normal_0, e) * jnp.dot(normal_0, normal_1)
                ) / A

                # Other attempt: https://math.stackexchange.com/a/4009935
                # Using crossproduct --> first one is prefered since it works in both dimensions
                # e = on_point_1 - on_point_0  # Order shouldn't matter?
                # n = jnp.cross(normal_0, normal_1)
                # param_0 = jnp.dot(jnp.cross(e, normal_1), n) / jnp.dot(n, n)
                # param_1 = jnp.dot(jnp.cross(e, normal_0), n) / jnp.dot(n, n)

                # Since we used the power of two for calculating the difference we need to consider all combinations here
                Ds = []
                points = []
                for sign_0, sign_1 in list(product([1.0, -1.0], repeat=2)):
                    point_0 = on_point_0 + sign_0 * param_0 * normal_0
                    point_1 = on_point_1 + sign_1 * param_1 * normal_1
                    D = point_0 - point_1
                    Ds.append(D)
                    points.append((point_0, point_1))
                    # print("Distance: {}".format(jnp.linalg.norm(D)))

                smallest_D_idx = jnp.argmin(jnp.linalg.norm(jnp.array(Ds), axis=1))
                D = Ds[smallest_D_idx]
                point_0, point_1 = points[smallest_D_idx]
                print("Direct Residual: {}".format(jnp.linalg.norm(D)))

                center_circle_ = point_0 + D / 2
            if frame == "svd":
                center_circle_svd = jnp.concatenate((center_circle_, jnp.zeros((1,))))
                center_circle_world = T_world_svd.apply(center_circle_svd)
            elif frame == "world":
                center_circle_world = center_circle_
                if visualize:
                    center_circle_svd = T_svd_world.apply(center_circle_world)
                    if not direct_degeneration:
                        on_point_0 = T_svd_world.apply(on_point_0)
                        on_point_1 = T_svd_world.apply(on_point_1)
                        normal_0 = T_svd_world.rotation().apply(normal_0)
                        normal_1 = T_svd_world.rotation().apply(normal_1)
                        point_0 = T_svd_world.apply(point_0)
                        point_1 = T_svd_world.apply(point_1)

        print("center_circle_world", center_circle_world)

        parameters = RevoluteJointParameters(
            base_transform=base_transform,
            center_of_rotation=center_circle_world,
            axis_of_rotation=rot / jnp.linalg.norm(rot),
        )

        if visualize:
            import matplotlib.pyplot as plt
            from visualizer import Arrow3D, Visualizer

            fig = plt.figure()
            ax1 = fig.add_subplot(221, projection="3d")

            # ax1
            ax1.set_title("World")
            ax1.set_autoscale_on(False)
            scatter_points = ax1.scatter(
                points_world[:, 0], points_world[:, 1], points_world[:, 2], color="blue"
            )
            # ax.scatter(center[0], center[1], center[2], color="green")
            arrow_rot = Arrow3D(
                [center_circle_world[0], center_circle_world[0] + rot[0]],
                [center_circle_world[1], center_circle_world[1] + rot[1]],
                [center_circle_world[2], center_circle_world[2] + rot[2]],
                color="red",
            )
            ax1.add_artist(arrow_rot)
            arrow_center_points = Arrow3D(
                [0, center_points_world[0]],
                [0, center_points_world[1]],
                [0, center_points_world[2]],
                color="green",
            )
            ax1.add_artist(arrow_center_points)
            arrow_center_world = Arrow3D(
                [0, center_circle_world[0]],
                [0, center_circle_world[1]],
                [0, center_circle_world[2]],
                color="blue",
            )
            ax1.add_artist(arrow_center_world)
            ax1.legend(
                [scatter_points, arrow_rot, arrow_center_points, arrow_center_world],
                [
                    "Points in World",
                    "Rotation Axis",
                    "Center of Points",
                    "Center of Circle",
                ],
            )

            # ax2
            ax2 = fig.add_subplot(223, projection="3d")
            ax2.set_title("SVD")
            ax2.set_autoscale_on(False)
            scatter_points_svd = ax2.scatter(
                points_svd[:, 0], points_svd[:, 1], points_svd[:, 2], color="orange"
            )
            points_trans = points_world - center_points_world
            scatter_points_trans = ax2.scatter(
                points_trans[:, 0], points_trans[:, 1], points_trans[:, 2], color="blue"
            )
            visu_ax2 = Visualizer(ax2)
            visu_ax2.add_frame(SO3.from_matrix(vv), "SVD")
            ax2.legend(
                [scatter_points_trans, scatter_points_svd],
                ["Points in World", "Points in SVD"],
            )

            # ax3
            # ax3 = fig.add_subplot(122)
            plt.figure()
            ax3 = plt.gca()
            ax3.set_aspect("equal")
            tmp1 = ax3.scatter(points_svd[:, 0], points_svd[:, 1], color="tab:blue")
            tmp2 = ax3.scatter(
                center_circle_svd[0], center_circle_svd[1], color="tab:red"
            )
            if method_center == "direct" and not direct_degeneration:
                normal_scale = max(param_0, param_1) * 2

                tmp3 = ax3.scatter(on_point_0[0], on_point_0[1], color="tab:green")
                start_points_0 = on_point_0 - normal_scale * normal_0
                end_points_0 = on_point_0 + normal_scale * normal_0
                ax3.plot(
                    [start_points_0[0], end_points_0[0]],
                    [start_points_0[1], end_points_0[1]],
                    color="tab:green",
                )
                ax3.plot(
                    [points_svd_2d[0, 0], points_svd_2d[1, 0]],
                    [points_svd_2d[0, 1], points_svd_2d[1, 1]],
                    color="tab:green",
                )

                tmp4 = ax3.scatter(on_point_1[0], on_point_1[1], color="tab:orange")
                start_points_1 = on_point_1 - normal_scale * normal_1
                end_points_1 = on_point_1 + normal_scale * normal_1
                ax3.plot(
                    [start_points_1[0], end_points_1[0]],
                    [start_points_1[1], end_points_1[1]],
                    color="tab:orange",
                )
                ax3.plot(
                    [points_svd_2d[0, 0], points_svd_2d[2, 0]],
                    [points_svd_2d[0, 1], points_svd_2d[2, 1]],
                    color="tab:orange",
                )

                # tmp5 = ax3.scatter(point_0[0], point_0[1])
                # tmp6 = ax3.scatter(point_1[0], point_1[1])

                tmp7 = ax3.add_patch(
                    plt.Circle(
                        (center_circle_svd[0], center_circle_svd[1]),
                        onp.linalg.norm(center_circle_svd - points_svd[0, :]),
                        fill=False,
                        color="red",
                        alpha=0.8,
                    )
                )

                ax3.legend(
                    [
                        tmp1,
                        tmp2,
                        tmp3,
                        tmp4,
                        # tmp5,
                        # tmp6,
                    ],
                    [
                        "Datapoints",
                        "Center Circle",
                        "Perpendicular Bisector 1",
                        "Perpendicular Bisector 2",
                        # "point 0",
                        # "point 1",
                    ],
                )

        return parameters

    @jax.jit
    @overrides
    def forward(parameters, joint_state: float):
        return (
            parameters.base_transform
            @ SE3.from_rotation_and_translation(
                rotation=SO3.identity(), translation=parameters.center_of_rotation
            )
            @ SE3.from_rotation_and_translation(
                rotation=SO3.exp(parameters.axis_of_rotation * joint_state),
                translation=jnp.zeros((3,)),
            )
            @ SE3.from_rotation_and_translation(
                rotation=SO3.identity(), translation=-parameters.center_of_rotation
            )
        )

    @jax.jit
    @overrides
    def backward(parameters, transformation: SE3) -> float:
        tangent_rot = (
            (
                SE3.from_rotation_and_translation(
                    rotation=SO3.identity(),
                    translation=-parameters.center_of_rotation,  ### Inverse: flip minus
                )
                @ parameters.base_transform.inverse()
                @ transformation
                @ SE3.from_rotation_and_translation(
                    rotation=SO3.identity(),
                    translation=parameters.center_of_rotation,  ### Inverse: flip minus
                )
            )
            .rotation()
            .log()
        )
        return jnp.dot(tangent_rot, parameters.axis_of_rotation)


#########
# TWIST #
#########
@jax_dataclasses.pytree_dataclass
class TwistJointParameters(BaseJointParameters):
    # Additional axis
    twist: jnp.ndarray

    def get_flat_parameter_vector(self):
        return jnp.concatenate((self.base_transform.log(), self.twist))

    @staticmethod
    def from_flat_parameter_vector(flat_vector):
        return RevoluteJointParameters(
            base_transform=SE3.exp(flat_vector[:6]),
            twist=flat_vector[6:],
        )

    @overrides
    def to_twist(self) -> jnp.ndarray:
        return self.twist


class TwistJoint(BaseJoint):
    min_samples = 3
    parameter_type = RevoluteJointParameters
    motion_type = helpers.MotionType.ROT

    @overrides
    def estimate_parameters(
        samples: List[SE3], visualize=False
    ) -> TwistJointParameters:
        raise NotImplementedError()
        return TwistJointParameters(base_transform=SE3.identity(), twist=jnp.zeros(6))

    @overrides
    # @jax.jit
    def forward(parameters, joint_state: float):
        return parameters.base_transform @ jaxlie.SE3.exp(
            parameters.twist * joint_state
        )

    @overrides
    # @jax.jit
    def backward(parameters, transformation: SE3) -> float:
        raise NotImplementedError()
        return 0.0
