import abc
import dataclasses
import traceback
from typing import ClassVar, List

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxfg
import jaxlie
from overrides import overrides

from typing import Any

import helpers


## Define some aliases
class LatentPoseVariable(jaxfg.geometry.SE3Variable):
    pass


class LatentTransformationVariable(jaxfg.geometry.SE3Variable):
    pass


class BaseTransformationVariable(jaxfg.geometry.SE3Variable):
    pass


@jax_dataclasses.pytree_dataclass
class JointParameters:
    params: jnp.ndarray


class JointParametersVariable(jaxfg.core.VariableBase[JointParameters]):
    @classmethod
    def get_random_value(self, key: Any) -> JointParameters:
        return jax.random.uniform(key, shape=self.get_default_value().params.shape)


class GeneralJointParametersVariable(JointParametersVariable):
    # Only needed if local parameter dim differs from the default parameter value
    # @classmethod
    # @overrides
    # def get_local_parameter_dim(self) -> int:
    #     return 6

    @classmethod
    @overrides
    def get_default_value(self) -> JointParameters:
        return JointParameters(params=jnp.ones((6,)))

    @classmethod
    @overrides
    def manifold_retract(
        self, x: JointParameters, local_delta: jaxfg.hints.Array
    ) -> JointParameters:
        return JointParameters(params=x.params + local_delta)


class DecoupledJointParametersVariable(JointParametersVariable):
    # Only needed if local parameter dim differs from the default parameter value
    # @classmethod
    # @overrides
    # def get_local_parameter_dim(self) -> int:
    #     return 8

    @classmethod
    @overrides
    def get_default_value(self) -> JointParameters:
        return JointParameters(params=jnp.ones((8,)))

    @classmethod
    @overrides
    def manifold_retract(
        cls, x: JointParameters, local_delta: jaxfg.hints.Array
    ) -> JointParameters:
        new_params_ = x.params + local_delta
        new_params = jnp.hstack([new_params_[0:6], jnp.sqrt(new_params_[6:8] ** 2)])
        normalization = jnp.hstack(
            [
                jnp.ones((3,)),
                jnp.full((3,), jnp.linalg.norm(new_params[3:6])),
                # jnp.full((2,), jnp.linalg.norm(new_params[6:8])),
                jnp.full((2,), jnp.sum(new_params[6:8])),
                # jnp.ones((2,)),
            ]
        )
        return JointParameters(params=new_params / normalization)


class PrismaticJointParametersVariable(JointParametersVariable):
    @classmethod
    @overrides
    def get_local_parameter_dim(self) -> int:
        return 3

    @classmethod
    @overrides
    def get_default_value(self) -> JointParameters:
        return JointParameters(params=jnp.hstack([jnp.ones((3,)), jnp.zeros((3,))]))

    @classmethod
    @overrides
    def manifold_retract(
        self, x: JointParameters, local_delta: jaxfg.hints.Array
    ) -> JointParameters:
        # Extract translation part from twist
        # jax.experimental.host_callback.id_print(local_delta)
        old_trans = x.params[:3]
        new_trans = (
            old_trans + local_delta
        )  # Ignore the three remaining entries, should be zero anyway
        new_trans /= jnp.linalg.norm(new_trans)
        return JointParameters(params=jnp.hstack([new_trans, jnp.zeros((3,))]))


class RevoluteJointParametersVariable(JointParametersVariable):
    # @classmethod
    # @overrides
    # def get_local_parameter_dim(self) -> int:
    #     return 6

    @classmethod
    @overrides
    def get_default_value(self) -> JointParameters:
        return JointParameters(params=jnp.ones((6,)))

    @classmethod
    @overrides
    def manifold_retract(
        self, x: JointParameters, local_delta: jaxfg.hints.Array
    ) -> JointParameters:
        new_twist_ = x.params + local_delta
        rot = new_twist_[3:] / jnp.linalg.norm(new_twist_[3:])
        # Enforce rotation by rejection --> Force rot
        pos = jnp.cross(rot, jnp.cross(new_twist_[:3], rot))
        new_twist = jnp.concatenate((pos, rot))
        return JointParameters(params=new_twist)


class ScrewNetJointParametersVariable(JointParametersVariable):
    # @classmethod
    # @overrides
    # def get_local_parameter_dim(self) -> int:
    #     return 6

    @classmethod
    @overrides
    def get_default_value(self) -> JointParameters:
        return JointParameters(params=jnp.ones((6,)))

    @classmethod
    @overrides
    def manifold_retract(
        self, x: JointParameters, local_delta: jaxfg.hints.Array
    ) -> JointParameters:
        new_twist_ = x.params + local_delta
        ori = new_twist_[3:] / jnp.linalg.norm(new_twist_[3:])  # l in ScrewNet
        mom = new_twist_[:3]  # m in ScrewNet
        # Enforce rotation by rejection --> Force cross product to be zero
        # <m, l> = 0
        pos = jnp.cross(ori, jnp.cross(mom, ori))
        new_twist = jnp.concatenate((pos, mom))
        return JointParameters(params=new_twist)


@jax_dataclasses.pytree_dataclass
class JointState:
    state: jnp.ndarray


class JointStateVariable(jaxfg.core.VariableBase[JointState]):
    @classmethod
    @overrides
    def get_local_parameter_dim(self) -> int:
        return 1

    @classmethod
    @overrides
    def get_default_value(self) -> JointState:
        return JointState(
            state=jnp.ones(
                self.get_local_parameter_dim(),
            )
            * 1e-0
        )

    @classmethod
    @overrides
    def manifold_retract(
        self, x: JointState, local_delta: jaxfg.hints.Array
    ) -> JointState:
        return JointState(state=x.state + local_delta)

    @classmethod
    def manifold_inverse_retract(self, x: JointState, y: JointState) -> jnp.ndarray:
        return x.state - y.state


class JointStateOneVariable(JointStateVariable):
    pass


class JointStateTwoVariable(JointStateVariable):
    @classmethod
    @overrides
    def get_local_parameter_dim(self) -> int:
        return 2
