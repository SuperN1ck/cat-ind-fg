import enum
from collections import defaultdict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax_dataclasses
import jaxfg
import jaxlie
import numpy as onp
from jaxfg import geometry
from sample_generator import JointConnection
from visualizer import visualize_graph
import copy

import helpers
from factor_graph import factors as factor_definitions
from factor_graph import helpers as fg_helpers
from factor_graph.states import (
    BaseTransformationVariable,
    JointParametersVariable,
    LatentPoseVariable,
    LatentTransformationVariable,
)
from helpers import (
    MotionType,
    clean_twist,
    get_motion_type_from_twist,
    RandomKeyGenerator,
)
import random

# VARIANCE_SCALE_EXP: float = 1e-5
VARIANCE_SCALE_EXP: float = 1e-10


class GraphOptions(NamedTuple):
    observe_part_poses: bool = True
    observe_transformation: bool = True
    observe_part_pose_betweens: bool = False
    observe_part_centers: bool = False
    latent_part_poses: bool = True
    joint_parameter_prior: bool = False
    seed_with_observations: bool = True
    decoupled_twist = False


# TODO Maybe refactor into one big observation factory
# - Keep track of factor indices
# - Generation method
# - Know how to handle updates
# - Create partial dicts for initilization


class Graph:
    graph: jaxfg.core.StackedFactorGraph
    graph_options: GraphOptions
    structure: Dict[str, JointConnection]
    observations: Dict[str, List[jaxlie.SE3]]
    obs_part_pose_factor_indices: List[str]
    obs_between_factor_indices: List[str]
    obs_center_factor_indices: List[str]
    latent_pose_variables: Dict[str, List[LatentPoseVariable]]
    joint_parameter_variables: Dict[str, JointParametersVariable]
    base_transformation_variables: Dict[str, BaseTransformationVariable]
    observation_variance: float
    joint_formulations: Dict[str, fg_helpers.JointFormulation]

    def __init__(self):
        self.clear()

    def clear(self):
        self.graph = None
        self.T = 0
        self.structure = {}
        self.observations = {}
        self.obs_part_pose_factor_indices = []
        self.obs_between_factor_indices = []
        self.obs_center_factor_indices = []
        self.latent_pose_variables = {}
        self.joint_parameter_variables = {}
        self.base_transformation_variables = {}
        self.observation_variance = 0.0
        self.joint_formulations = {}
        self.huber_delta = None

    def build_graph(
        self,
        T,
        structure: Dict[str, JointConnection],
        factor_graph_options: GraphOptions,
        joint_formulations: Dict[str, fg_helpers.JointFormulation] = defaultdict(
            lambda: fg_helpers.JointFormulation.GeneralTwist
        ),
        variance_exp_factor=jnp.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]),
        all_hubers=False,
        huber_delta=None,
        visualize=False,
    ):
        ## Create Noise Models

        observation_noise_model = jaxfg.noises.DiagonalGaussian.make_from_covariance(
            jnp.ones(6)
        )
        if all_hubers:
            observation_noise_model = jaxfg.noises.HuberWrapper(
                wrapped=observation_noise_model,  # Will be replaced later
                delta=jnp.array([1.35]),  # Will be replaced later
            )

        assert variance_exp_factor.shape == (6,)
        joint_noise_model = jaxfg.noises.DiagonalGaussian.make_from_covariance(
            jnp.array(variance_exp_factor)
        )
        if huber_delta:
            HUBER_DELTA = huber_delta
        else:
            HUBER_DELTA = 1 / jnp.linalg.norm(variance_exp_factor)
        print(f"{HUBER_DELTA = }")
        joint_noise_model = jaxfg.noises.HuberWrapper(
            wrapped=joint_noise_model, delta=HUBER_DELTA
        )

        center_noise_model = jaxfg.noises.DiagonalGaussian.make_from_covariance(
            jnp.ones(3)
        )
        if all_hubers:
            center_noise_model = jaxfg.noises.HuberWrapper(
                wrapped=center_noise_model,  # Will be replaced later
                delta=jnp.array([1.35]),  # Will be replaced later
            )

        self.huber_delta = huber_delta

        ## Keep track of latent variables
        factors: List[jaxfg.core.FactorBase] = []

        def add_pose_observation_factor(latent_pose_variables, id):
            for pose_lat in latent_pose_variables:
                factors.append(
                    factor_definitions.PoseObservationFactor.make(
                        observation=jaxlie.SE3.identity(),  # Will be replaced later
                        variable=pose_lat,
                        noise_model=observation_noise_model,  # Will be replaced later
                    )
                )
            # Keep track of order observations were added
            self.obs_part_pose_factor_indices.append(id)

        def add_between_observation_factor(latent_pose_variables, id):
            for variable_before, variable_after in zip(
                latent_pose_variables[:-1], latent_pose_variables[1:]
            ):
                factors.append(
                    factor_definitions.BetweenFactor.make(
                        variable_T_world_a=variable_before,
                        variable_T_world_b=variable_after,
                        T_a_b=jaxlie.SE3.identity(),  # Will be replaced later
                        noise_model=observation_noise_model,  # Will be replaced later
                    )
                )
            self.obs_between_factor_indices.append(id)

        def add_center_observation_factor(latent_pose_variables, id):
            for pose_lat in latent_pose_variables:
                factors.append(
                    factor_definitions.TranslationObservationFactor.make(
                        observation=jnp.zeros((3,)),  # Will be replaced later
                        variable=pose_lat,
                        noise_model=center_noise_model,  # Will be replaced later
                    )
                )
            # Keep track of ordered observations were added
            self.obs_center_factor_indices.append(id)

        for joint_id, joint_connection in structure.items():
            # Check whether we already parsed the two variables
            # ------ from part -------
            latent_pose_variables_from, exists = self.get_latent_pose_variables(
                joint_connection.from_id, T=T, create=True
            )
            if not exists:
                if factor_graph_options.observe_part_poses:
                    add_pose_observation_factor(
                        latent_pose_variables_from, joint_connection.from_id
                    )
                if factor_graph_options.observe_part_pose_betweens:
                    add_between_observation_factor(
                        latent_pose_variables_from, joint_connection.from_id
                    )
                if factor_graph_options.observe_part_centers:
                    add_center_observation_factor(
                        latent_pose_variables_from, joint_connection.from_id
                    )
            # ------ to part -------
            latent_pose_variables_to, exists = self.get_latent_pose_variables(
                joint_connection.to_id, T=T, create=True
            )
            if not exists:
                if factor_graph_options.observe_part_poses:
                    add_pose_observation_factor(
                        latent_pose_variables_to, joint_connection.to_id
                    )
                if factor_graph_options.observe_part_pose_betweens:
                    add_between_observation_factor(
                        latent_pose_variables_to, joint_connection.to_id
                    )
                if factor_graph_options.observe_part_centers:
                    add_center_observation_factor(
                        latent_pose_variables_to, joint_connection.to_id
                    )
            # ----------------------
            joint_formulation = joint_formulations[joint_id]

            # Create the exponent factor
            joint_parameters_variable = fg_helpers.JointGraphVariableMapping[
                joint_formulation
            ].parameters_variable()
            print(f"{type(joint_parameters_variable) = }")

            self.joint_parameter_variables[joint_id] = joint_parameters_variable
            base_transformation_variable = BaseTransformationVariable()
            self.base_transformation_variables[joint_id] = base_transformation_variable

            joint_state_variables = [
                fg_helpers.JointGraphVariableMapping[joint_formulation].state_variable()
                for _ in range(T)
            ]
            for from_lat, to_lat, joint_state in zip(
                latent_pose_variables_from,
                latent_pose_variables_to,
                joint_state_variables,
            ):
                factors.append(
                    fg_helpers.JointGraphVariableMapping[joint_formulation].factor.make(
                        from_pose=from_lat,
                        to_pose=to_lat,
                        base_transformation_variable=base_transformation_variable,
                        joint_parameters_variable=joint_parameters_variable,
                        joint_state_variable=joint_state,
                        noise_model=joint_noise_model,
                    )
                )

            # TODO Maybe add an in between factor for joint state variables
            # --> to ensure temporal consistency

        self.graph = jaxfg.core.StackedFactorGraph.make(factors)

        if visualize:
            self.graph_vis = visualize_graph(factors)

        self.structure = structure
        self.T = T
        self.graph_options = factor_graph_options
        self.joint_formulations = joint_formulations

    def get_latent_pose_variables(
        self,
        key,
        T=1,
        create=False,
        variable_type=LatentPoseVariable,
    ):
        exists = key in self.latent_pose_variables.keys()
        if exists:
            return self.latent_pose_variables[key], True
        elif create:
            latent_pose_variables = [variable_type() for _ in range(T)]
            self.latent_pose_variables[key] = latent_pose_variables
            return latent_pose_variables, False
        else:
            return None, False

    def update_variables(
        self,
        new_values,
        new_variance: Union[Dict[str, onp.ndarray], float, onp.ndarray, jnp.ndarray],
        noise_dim,
        indices_to_update,
        update_fun,
        factor_type,
        use_huber=False,
    ):
        assert all(index in new_values for index in indices_to_update)

        ordered_values = [
            val for id in indices_to_update for val in new_values[id]
        ]  # Just in case order observations as before

        new_values_stacked = jax.tree_map(  # Flatten them before applying
            lambda *leaves: jnp.vstack(leaves), *ordered_values
        )

        if isinstance(new_variance, float):
            new_variance = new_variance * jnp.ones(noise_dim)

        assert new_variance.shape[-1] == noise_dim
        new_noises = [
            jaxfg.noises.HuberWrapper(
                wrapped=jaxfg.noises.DiagonalGaussian.make_from_covariance(
                    new_variance
                ),
                delta=self.huber_delta
                if self.huber_delta
                else 1 / jnp.linalg.norm(new_variance),
            )
            if use_huber
            else jaxfg.noises.DiagonalGaussian.make_from_covariance(new_variance)
        ] * len(ordered_values)

        new_noises_stacked = jax.tree_map(  # Flatten them before applying
            lambda *leaves: jnp.vstack(leaves),
            *new_noises,
        )

        with jax_dataclasses.copy_and_mutate(self.graph) as new_graph:
            for factor_stack in new_graph.factor_stacks:
                # Skim through all factors
                if not isinstance(factor_stack.factor, factor_type):
                    continue

                factor_stack.factor = update_fun(
                    factor_stack.factor, new_values_stacked, new_noises_stacked
                )

                # No need to loop through the remaining factor stacks
                break

        self.graph = new_graph

    def update_betweens(self, new_betweens, new_variance, use_huber=False):
        def update_fun(
            factor: factor_definitions.BetweenFactor,
            new_betweens_stacked,
            new_noises_stacked,
        ):
            factor.T_a_b = new_betweens_stacked
            factor.noise_model = new_noises_stacked
            return factor

        self.update_variables(
            new_betweens,
            new_variance,
            6,
            self.obs_between_factor_indices,
            update_fun,
            factor_definitions.BetweenFactor,
            use_huber=use_huber,
        )

    def update_poses(self, new_observations, new_variance, use_huber=False):
        def update_fun(
            factor: factor_definitions.PoseObservationFactor,
            new_observations_stacked,
            new_noises_stacked,
        ):
            factor.observation = new_observations_stacked
            factor.noise_model = new_noises_stacked
            return factor

        self.update_variables(
            new_observations,
            new_variance,
            6,
            self.obs_part_pose_factor_indices,
            update_fun,
            factor_definitions.PoseObservationFactor,
            use_huber=use_huber,
        )

        self.observation_variance = new_variance
        self.observations = new_observations

    def update_centers(self, new_centers, new_variance: float, use_huber=False):
        def update_fun(
            factor: factor_definitions.TranslationObservationFactor,
            new_centers_stacked,
            new_noises_stacked,
        ):
            factor.observation = new_centers_stacked
            factor.noise_model = new_noises_stacked
            return factor

        self.update_variables(
            new_centers,
            new_variance,
            3,
            self.obs_center_factor_indices,
            update_fun,
            factor_definitions.TranslationObservationFactor,
            use_huber=use_huber,
        )

    def solve_graph(
        self,
        initial_base_transform: Optional[
            jaxlie.SE3
        ] = None,  # TODO extend to multiple joints
        initial_parameters: Optional[
            jnp.ndarray
        ] = None,  # TODO also extend to multiple joints
        max_restarts: int = 100,
        cost_threshold: float = 1e12,
        aux_data_in={},
    ):
        aux_data = aux_data_in.copy()
        # # Build a dict for feeding in our initial assignment TODO Make this iterable
        # partially_assign = {}
        # if initial_parameters is not None:
        #     partially_assign[
        #         list(self.joint_parameter_variables.values())[0]
        #     ] = initial_parameters
        # if initial_base_transform is not None:
        #     partially_assign[
        #         list(self.base_transformation_variables.values())[0]
        #     ] = initial_base_transform

        def get_partial_assign():
            partially_assign = {}
            for joint_parameter_variable in list(
                self.joint_parameter_variables.values()
            ):
                partially_assign[
                    joint_parameter_variable
                ] = joint_parameter_variable.get_random_value(
                    RandomKeyGenerator.single_key(random.getrandbits(32))
                )
            # Assign random base transform
            for base_transformation_variable in list(
                self.base_transformation_variables.values()
            ):
                partially_assign[
                    base_transformation_variable
                ] = jaxlie.SE3.sample_uniform(
                    RandomKeyGenerator.single_key(random.getrandbits(32))
                )
            # Seed with observation variables
            if (
                self.graph_options.seed_with_observations
                and self.observations  # Check for non empty!
            ):
                for joint_connection in self.structure.values():
                    for pose_id in {
                        joint_connection.from_id,
                        joint_connection.to_id,
                        joint_connection.via_id,
                    }:
                        if not pose_id in self.observations.keys():
                            continue

                        variable_list, exists = self.get_latent_pose_variables(
                            pose_id, create=False
                        )
                        if not exists:
                            continue

                        for variable, obs_value in zip(
                            variable_list, self.observations[pose_id]
                        ):
                            partially_assign[variable] = obs_value
            return partially_assign

        ## Create a list of initial partial assigns
        initial_assignments = []
        for _ in range(max_restarts):
            partial_assign = get_partial_assign()
            initial_assignment = jaxfg.core.VariableAssignments.make_from_partial_dict(
                self.graph.get_variables(), partial_assign
            )
            initial_assignments.append(initial_assignment)

        use_vmap = self.T < 100
        if use_vmap:
            print("Using VMAP to optimize all factor graph version")
            stacked_assignments, all_costs = fg_helpers.solve_multiple_times(
                self.graph, initial_assignments, max_restarts, use_vmap=True
            )
            idx_lowest_cost = jnp.nanargmin(all_costs)
            best_costs = all_costs[idx_lowest_cost]
            best_assignment = helpers.get_sample(stacked_assignments, idx_lowest_cost)
        else:
            print("Using direct version to optimize over factor graph")
            best_costs = jnp.inf
            best_assignment = None
            all_costs = []
            for idx, initial_assignment in enumerate(initial_assignments):
                solution_assignment, costs = fg_helpers.solve_single(
                    initial_assignment, graph=self.graph
                )
                print(f"[{idx}] Costs {costs = }")
                all_costs.append(costs)
                if costs < best_costs:
                    best_costs = costs
                    best_assignment = solution_assignment
            all_costs = onp.array(all_costs)

        print(f"{all_costs = }\n{best_costs = }")

        # TODO Make iterable for a full graph setup, only one joint parameter variable currently
        # Values is confusing here, and refers to values of the dict, returning a variable
        joint_formulation = list(self.joint_formulations.values())[0]
        joint_parameters = best_assignment.get_value(
            list(self.joint_parameter_variables.values())[0]
        ).params
        twist = fg_helpers.get_twist_from_parameters(
            joint_parameters,
            joint_formulation,
        )
        base_transform = best_assignment.get_value(
            list(
                self.base_transformation_variables.values()  # Values is confusing here, and refers to values of the dict
            )[0]
        )

        if "best_assignment" in aux_data.keys():
            aux_data["best_assignment"] = best_assignment
        if "parameters" in aux_data.keys():
            aux_data["parameters"] = joint_parameters
        if "joint_states" in aux_data.keys():
            joint_state_variable_type = fg_helpers.JointGraphVariableMapping[
                joint_formulation
            ].state_variable
            joint_states = best_assignment.get_stacked_value(joint_state_variable_type)
            aux_data["joint_states"] = joint_states.state
        if "latent_poses" in aux_data.keys():
            latent_poses_stacked = best_assignment.get_stacked_value(LatentPoseVariable)
            # Most hacky way to unstack stuff? --> Should consider structure
            latent_poses_list = [
                jax.tree_map(lambda leaf: leaf[i], latent_poses_stacked)
                for i in range(2 * self.T)
            ]
            latent_poses_named = {
                "first": latent_poses_list[: self.T],
                "second": latent_poses_list[self.T :],
            }
            aux_data["latent_poses"] = latent_poses_named

        # TODO run a refinement step, where we takes this tiwst and run a constrained optimization on the graph?

        return twist, base_transform, aux_data
