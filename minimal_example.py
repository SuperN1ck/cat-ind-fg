import argparse
import factor_graph
import jax.numpy as jnp
import jaxlie
import numpy as onp
from sample_generator import JointConnection

def get_SE3_pose(pos, ori):
    assert pos.shape == (3,)
    assert ori.shape == (4,)
    return jaxlie.SE3.from_rotation_and_translation(
        translation=jnp.array(pos), rotation=jaxlie.SO3.from_quaternion_xyzw(ori)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", default=None, help="""Filepath""")
    args = parser.parse_args()

    data = onp.load(args.file)
    pos = data["pos"]  # (T, 3)
    ori = data["ori"]  # (T, 4); XYZW

    poses = [get_SE3_pose(pos, ori) for pos, ori in zip(pos, ori)]

    # Build the graph
    graph = factor_graph.graph.Graph()

    factor_graph_options = factor_graph.graph.GraphOptions(
        observe_transformation=True,
        observe_part_poses=False,
        observe_part_pose_betweens=False,
        observe_part_centers=False,
        seed_with_observations=False,
    )

    structure = {
        "first_second": JointConnection(
            from_id="first", to_id="second", via_id="first_second"
        )
    }

    joint_formulations = {
        "first_second": factor_graph.helpers.JointFormulation.GeneralTwist
    }

    graph.build_graph(
        pos.shape[0],  # Amount of time steps
        structure,
        factor_graph_options,
        joint_formulations,
    )
    graph.update_poses({"first_second": poses}, 1e-8)
    graph.solve_graph()
