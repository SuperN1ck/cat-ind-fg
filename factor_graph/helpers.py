import enum
from typing import Dict, NamedTuple

import helpers

import functools

import jax
from jax._src.numpy.lax_numpy import full
import jax.numpy as jnp
import numpy as onp

import jaxfg

import factor_graph

from helpers import normalize_twist


class JointFormulation(enum.Enum):
    # Non joint-specific tywpes
    GeneralTwist = enum.auto()
    # PitchEstimation = enum.auto()
    DecoupledTwist = enum.auto()
    ScrewNet = enum.auto()
    # Joint specific types
    Prismatic = enum.auto()
    Revolute = enum.auto()
    Rigid = enum.auto()


class JointModelGraphDefinitions(NamedTuple):
    parameters_variable: factor_graph.states.JointParametersVariable
    state_variable: factor_graph.states.JointStateVariable
    factor: factor_graph.factors.JointFactor


JointGraphVariableMapping: Dict[JointFormulation, JointModelGraphDefinitions] = {
    JointFormulation.GeneralTwist: JointModelGraphDefinitions(
        parameters_variable=factor_graph.states.GeneralJointParametersVariable,
        state_variable=factor_graph.states.JointStateOneVariable,
        factor=factor_graph.factors.ExponentialFactor,
    ),
    JointFormulation.Prismatic: JointModelGraphDefinitions(
        parameters_variable=factor_graph.states.PrismaticJointParametersVariable,
        state_variable=factor_graph.states.JointStateOneVariable,
        factor=factor_graph.factors.ExponentialFactor,
    ),
    JointFormulation.Revolute: JointModelGraphDefinitions(
        parameters_variable=factor_graph.states.RevoluteJointParametersVariable,
        state_variable=factor_graph.states.JointStateOneVariable,
        factor=factor_graph.factors.ExponentialFactor,
    ),
    JointFormulation.DecoupledTwist: JointModelGraphDefinitions(
        parameters_variable=factor_graph.states.DecoupledJointParametersVariable,
        state_variable=factor_graph.states.JointStateOneVariable,
        factor=factor_graph.factors.DecoupledExponentialFactor,
    ),
    JointFormulation.ScrewNet: JointModelGraphDefinitions(
        parameters_variable=factor_graph.states.ScrewNetJointParametersVariable,
        state_variable=factor_graph.states.JointStateTwoVariable,
        factor=factor_graph.factors.ScrewNetJointExponentialFactor,
    ),
}


def decoupled_twist_to_general(parameters):
    print(parameters)
    position = parameters[:3]
    orientation = parameters[3:6]
    rotation_portion = parameters[6]
    translation_portion = parameters[7]
    print(f"{rotation_portion = } {translation_portion = }")
    threshold = 1e-1

    twist_full = jnp.concatenate(
        (
            rotation_portion * (jnp.cross(-orientation, position))
            + translation_portion * orientation,
            rotation_portion * orientation,
        )
    )

    if (
        onp.abs(rotation_portion) < threshold
        and onp.abs(translation_portion) > threshold
    ):
        print("Translation")
        twist_pred = jnp.concatenate(
            (
                orientation,
                jnp.zeros(
                    3,
                ),
            )
        )
    elif (
        onp.abs(rotation_portion) > threshold
        and onp.abs(translation_portion) < threshold
    ):
        print("Rotation")
        twist_pred = jnp.concatenate((-jnp.cross(orientation, position), orientation))
    else:
        print("Helic")
        twist_pred = jnp.concatenate((-jnp.cross(orientation, position), orientation))

    print(f"{twist_full = }")
    print(f"{twist_pred = }")

    twist = twist_full
    # twist = twist_pred
    # return twist_pred
    # return full_twist
    return normalize_twist(twist)


def get_twist_from_parameters(parameters, joint_model_to_use):
    print(f"{joint_model_to_use = }")
    if joint_model_to_use == JointFormulation.DecoupledTwist:
        print(f"Norm for Orientation: {onp.linalg.norm(parameters[3:6])}")
        print(f"Norm for Portions: {onp.linalg.norm(parameters[6:8])}")

        return decoupled_twist_to_general(parameters)
    # TODO Implement pitch estimation?
    else:  # General, Prismatic, Revolute
        return parameters


def print_factor_residuals(factor_stacks, best_assignment):
    for stacked_factor in factor_stacks:
        residual_vector = stacked_factor.compute_residual_vector(best_assignment)
        residual_vector_whitened = jax.vmap(
            type(stacked_factor.factor.noise_model).whiten_residual_vector
        )(
            stacked_factor.factor.noise_model,
            residual_vector,
        )
        print(f"{type(stacked_factor.factor).__name__}")
        print(f"\t{jnp.sum(residual_vector ** 2) = } with {residual_vector.shape = }")
        print(f"\t{jnp.sum(residual_vector ** 2, axis=-1) = }")
        print(
            f"\t{jnp.sum(residual_vector_whitened ** 2) = } with"
            f" {residual_vector_whitened.shape = }"
        )
        print(f"\t{jnp.sum(residual_vector_whitened ** 2, axis=-1) = }")


@jax.jit
def solve_single(initial_assignment, graph=None):
    solution_assignment = graph.solve(
        initial_assignment,
        # solver=jaxfg.solvers.FixedIterationGaussNewtonSolver(
        # solver=jaxfg.solvers.GaussNewtonSolver(
        # solver=jaxfg.solvers.DoglegSolver( # Use this one without huber losses
        solver=jaxfg.solvers.LevenbergMarquardtSolver(  # Use this one with huber losses
            linear_solver=jaxfg.sparse.ConjugateGradientSolver(),
            # linear_solver=jaxfg.sparse.CholmodSolver(),
            # max_iterations=1000,
            verbose=False,
        ),
    )
    # print(solution_assignment)
    costs, _ = graph.compute_cost(solution_assignment)
    return solution_assignment, costs


@jax.jit
def loop_function(carry, assignment):
    best_costs, best_assignment, graph = carry
    solution_assignment, costs = solve_single(assignment, graph=graph)
    best_costs, best_assignment = jax.lax.cond(
        costs < best_costs,
        lambda _: (costs, solution_assignment),
        lambda _: (best_costs, best_assignment),
        operand=None,
    )
    return (best_costs, best_assignment, graph), costs


@functools.partial(jax.jit, static_argnums=(2, 3))
def solve_multiple_times(graph, initial_assignments, max_restarts, use_vmap=True):
    stacked_assignments = helpers.batch_samples(initial_assignments)

    if use_vmap:
        solution_assignments, all_costs = jax.vmap(
            functools.partial(solve_single, graph=graph)
        )(stacked_assignments)
        return solution_assignments, all_costs
    else:
        # Not used
        first_costs = jnp.inf
        first_assignment = initial_assignments[0]
        # # Old Loop Implementation
        # best_costs, best_assignment = jax.lax.fori_loop(
        #     0, max_restarts, loop_function, (first_costs, first_assignment)
        # )
        (best_costs, best_assignment, _), all_costs = jax.lax.scan(
            loop_function,
            (first_costs, first_assignment, graph),
            stacked_assignments,
            # unroll=5,
        )
        return best_assignment, best_costs, all_costs
