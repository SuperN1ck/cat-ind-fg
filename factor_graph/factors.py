import abc
import dataclasses
from typing import NamedTuple, Tuple
import jax

import jax.numpy as jnp
import jax_dataclasses
import jaxfg
import jaxlie
import numpy as onp
from baseline.joints import TwistJoint, TwistJointParameters
from overrides import overrides

import factor_graph.states as states
import helpers

LatentPoseObservationTuple = Tuple[jaxlie.SE3]

### Observe just the translation
@jax_dataclasses.pytree_dataclass
class TranslationObservationFactor(jaxfg.core.FactorBase[LatentPoseObservationTuple]):
    observation: jnp.ndarray

    @staticmethod
    def make(
        observation: jnp.ndarray,
        variable: states.LatentPoseVariable,
        noise_model: jaxfg.noises.NoiseModelBase,
    ) -> "TranslationObservationFactor":
        assert observation.shape == (3,)
        return TranslationObservationFactor(
            observation=observation, variables=(variable,), noise_model=noise_model
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: LatentPoseObservationTuple
    ) -> jnp.ndarray:
        value: jaxlie.SE3
        (value,) = variable_values
        return self.observation - value.translation()


### Observe the full pose
@jax_dataclasses.pytree_dataclass
class PoseObservationFactor(jaxfg.core.FactorBase[LatentPoseObservationTuple]):
    observation: jaxlie.SE3

    @staticmethod
    def make(
        observation: jaxlie.SE3,
        variable: states.LatentPoseVariable,
        noise_model: jaxfg.noises.NoiseModelBase,
    ) -> "PoseObservationFactor":
        return PoseObservationFactor(
            observation=observation,
            variables=(variable,),
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: LatentPoseObservationTuple
    ) -> jnp.ndarray:
        value: jaxlie.SE3
        (value,) = variable_values
        return jaxlie.manifold.rminus(value, self.observation)


class TransformationFactorValueTuple(NamedTuple):
    from_pose_value: jaxlie.SE3
    to_pose_value: jaxlie.SE3
    transformation_value: jaxlie.SE3


@jax_dataclasses.pytree_dataclass
class TransformationFactor(jaxfg.core.FactorBase[TransformationFactorValueTuple]):
    @staticmethod
    def make(
        from_pose_variable: states.LatentPoseVariable,
        to_pose_variable: states.LatentPoseVariable,
        transformation_variable: states.LatentTransformationVariable,
        noise_model: jaxfg.noises.NoiseModelBase,
    ) -> "TransformationFactor":
        return TransformationFactor(
            variables=(
                from_pose_variable,
                to_pose_variable,
                transformation_variable,
            ),
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: TransformationFactorValueTuple
    ) -> jnp.ndarray:
        from_pose_value = variable_values.from_pose_value
        to_pose_value = variable_values.to_pose_value
        transformation_value = variable_values.transformation_value
        return jaxlie.manifold.rminus(
            from_pose_value.inverse() @ to_pose_value, transformation_value
        )


class JointFactorValueTuple(NamedTuple):
    from_pose_value: jaxlie.SE3
    to_pose_value: jaxlie.SE3
    base_transformation_value: jaxlie.SE3
    joint_parameters_value: states.JointParameters
    joint_state_value: states.JointState


@jax_dataclasses.pytree_dataclass
class JointFactor(jaxfg.core.FactorBase[JointFactorValueTuple]):
    @classmethod
    def make(
        factor_class: "JointFactor",
        from_pose: states.LatentPoseVariable,
        to_pose: states.LatentPoseVariable,
        base_transformation_variable: states.BaseTransformationVariable,
        joint_parameters_variable: states.JointParametersVariable,
        joint_state_variable: states.JointStateVariable,
        noise_model: jaxfg.noises.NoiseModelBase,
    ) -> "JointFactor":
        return factor_class(
            variables=(
                from_pose,
                to_pose,
                base_transformation_variable,
                joint_parameters_variable,
                joint_state_variable,
            ),
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: JointFactorValueTuple
    ) -> jnp.ndarray:
        from_pose_value = variable_values.from_pose_value
        to_pose_value = variable_values.to_pose_value
        base_transformation_value = variable_values.base_transformation_value
        joint_parameters_value = variable_values.joint_parameters_value
        joint_state_value = variable_values.joint_state_value

        # TODO Use twist joint type class
        # twist_parameters = TwistJointParameters(
        #     base_transform=base_transformation_value, twist=joint_parameters_value.twist
        # )

        # return jaxlie.manifold.rminus(
        #     TwistJoint.forward(twist_parameters, joint_state_value.state[0]),
        #     transformation_value,
        # )

        joint_transformation = self.compute_joint_transformation(
            joint_parameters_value.params, joint_state_value.state
        )

        # return jaxlie.manifold.rminus(
        #     to_pose_value,
        #     from_pose_value @ base_transformation_value @ joint_transformation,
        # )
        return jaxlie.manifold.rminus(
            base_transformation_value @ joint_transformation,
            from_pose_value.inverse() @ to_pose_value,
        )

    @abc.abstractmethod
    def compute_joint_transformation(
        self, joint_parameters: jnp.ndarray, joint_state: jnp.ndarray
    ) -> jaxlie.SE3:
        """
        Compute joint specific transformation
        """
        raise NotImplementedError("Use class specific subclasses")


@jax_dataclasses.pytree_dataclass
class ExponentialFactor(JointFactor):
    @overrides
    def compute_joint_transformation(
        self, joint_parameters: jnp.ndarray, joint_state: jnp.ndarray
    ) -> jaxlie.SE3:
        assert joint_parameters.shape == (6,)
        assert joint_state.shape == (1,)
        return jaxlie.SE3.exp(joint_parameters * joint_state[0])


@jax_dataclasses.pytree_dataclass
class ScrewNetJointExponentialFactor(JointFactor):
    @overrides
    def compute_joint_transformation(
        self, joint_parameters: jnp.ndarray, joint_state: jnp.ndarray
    ) -> jaxlie.SE3:
        assert joint_parameters.shape == (6,)
        assert joint_state.shape == (2,)

        t = joint_state[0]  # translation
        theta = joint_state[1]  # rotation

        mom = joint_parameters[:3]
        ori = joint_parameters[3:]

        # This seems wrong?
        scaled_twist = jnp.concatenate((-theta * mom + t * ori, theta * ori))
        return jaxlie.SE3.exp(scaled_twist)


@jax_dataclasses.pytree_dataclass
class DecoupledExponentialFactor(JointFactor):
    @overrides
    def compute_joint_transformation(
        self, joint_parameters: jnp.ndarray, joint_state: jnp.ndarray
    ) -> jaxlie.SE3:
        assert joint_parameters.shape == (8,)
        assert joint_state.shape == (1,)

        pos = joint_parameters[:3]  # point on axis
        ori = joint_parameters[3:6]  # axis orientation
        g = joint_parameters[6]  # rotation portion
        l = joint_parameters[7]  # translation portion

        # twist = jnp.concatenate(
        #     (
        #         g * (jnp.cross(-ori * joint_state[0], pos)) + l * joint_state[0] * ori,
        #         g * ori * joint_state[0],
        #     )
        # )

        unscaled_twist = jnp.concatenate(
            (
                g * (jnp.cross(-ori, pos)) + l * ori,
                g * ori,
            )
        )
        twist = joint_state[0] * unscaled_twist

        return jaxlie.SE3.exp(twist)


###### Copied from jaxfg, but modified the residual function (see below) #######
class BetweenValueTuple(NamedTuple):
    T_world_a: jaxlie.SE3
    T_world_b: jaxlie.SE3


@jax_dataclasses.pytree_dataclass
class BetweenFactor(jaxfg.core.FactorBase[BetweenValueTuple]):
    T_a_b: jaxlie.SE3

    @staticmethod
    def make(
        variable_T_world_a: states.LatentPoseVariable,
        variable_T_world_b: states.LatentPoseVariable,
        T_a_b: jaxlie.SE3,
        noise_model: jaxfg.noises.NoiseModelBase,
    ) -> "BetweenFactor":
        assert type(variable_T_world_a) is type(variable_T_world_b)
        assert variable_T_world_a.get_group_type() is type(T_a_b)

        return BetweenFactor(
            variables=(
                variable_T_world_a,
                variable_T_world_b,
            ),
            T_a_b=T_a_b,
            noise_model=noise_model,
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: BetweenValueTuple
    ) -> jnp.ndarray:
        T_world_a = variable_values.T_world_a
        T_world_b = variable_values.T_world_b

        # Equivalent to: return ((T_world_a @ self.T_a_b).inverse() @ T_world_b).log()
        # return jaxlie.manifold.rminus(T_world_a @ self.T_a_b, T_world_b)
        # Modified to our setup:
        return jaxlie.manifold.rminus(self.T_a_b @ T_world_a, T_world_b)


###################### TODO Not further followed up on #########################
UnitTwistValueTuple = Tuple[states.JointParameters]


@jax_dataclasses.pytree_dataclass
class JointParameterPriorFactor(jaxfg.core.FactorBase[UnitTwistValueTuple]):
    motion_type_num: onp.ndarray  ## TODO make this helpers.MotionType once jaxfg#PR#3 is merged
    mu: jaxlie.SE3

    @staticmethod
    def make(
        variable: states.JointParametersVariable, motion_type: helpers.MotionType
    ) -> "JointParameterPriorFactor":
        noise = jaxfg.noises.DiagonalGaussian.make_from_covariance(
            diagonal=jnp.array([0.0, 0.0]),
        )
        return JointParameterPriorFactor(
            variables=(variable,),
            noise_model=noise,
            motion_type_num=onp.array([helpers.numeric_mapping[motion_type]]),
            mu=jaxlie.SE3.identity(),
        )

    @overrides
    def compute_residual_vector(
        self, variable_values: UnitTwistValueTuple
    ) -> jnp.ndarray:
        (twist_variable,) = variable_values
        twist = twist_variable.twist
        # TODO Maybe a better way is to slowly increase the importance of that residual

        # Unit
        # return jnp.where(
        #     self.motion_type_num[0] == helpers.numeric_mapping[helpers.MotionType.RIGID],
        #     jnp.array(
        #         [jnp.linalg.norm(twist[:3]), jnp.linalg.norm(twist[3:])]
        #     ),  # RIGID, trans and rot should stay 0
        #     jnp.where(
        #         self.motion_type_num[0]
        #         == helpers.numeric_mapping[helpers.MotionType.TRANS],
        #         jnp.array(
        #             [1 - jnp.linalg.norm(twist[:3]), jnp.linalg.norm(twist[3:])]
        #         ),  # TRANS, trans should become 1.
        #         jnp.array([0.0, 1 - jnp.linalg.norm(twist[3:])]),  # Rot/Helic
        #     ),
        # )

        # Non-Unit
        return jnp.where(
            self.motion_type_num[0]
            == helpers.numeric_mapping[helpers.MotionType.RIGID],
            jnp.array(
                [
                    jnp.linalg.norm(twist[:3]),
                    jnp.linalg.norm(twist[3:]),
                ]
            ),  # RIGID, trans and rot should stay 0
            jnp.where(
                self.motion_type_num[0]
                == helpers.numeric_mapping[helpers.MotionType.TRANS],
                jnp.array(
                    [0.0, jnp.linalg.norm(twist[3:])]
                ),  # TRANS, trans doesn't matter, rot should stay 0
                jnp.array([0.0, 0.0]),  # Rot/Helic, nothing matters
            ),
        )
