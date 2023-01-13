import enum
from typing import Any, List

import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

sns.set()

import jax.numpy as jnp
import numpy as onp
import open3d
from jaxlie import SE3
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import jaxlie

import helpers
from helpers import MotionType


def get_twist_axis_points(twist_, frame):
    axises = []
    axis_scale = 4.0  # How to define that scaling?
    twist = helpers.normalize_twist(helpers.transform_twist_rel(twist_, frame))

    trans = twist[:3]
    rot = twist[3:]

    # Determine type of motion
    motion_type = helpers.get_motion_type_from_twist(twist)

    if motion_type == MotionType.TRANS:  # Equal to an infinity pitch
        # We have a pure translational joint
        long_trans_axis = trans * axis_scale
        axis = [
            [
                -long_trans_axis[0] + frame.translation()[0],
                long_trans_axis[0] + frame.translation()[0],
            ],
            [
                -long_trans_axis[1] + frame.translation()[1],
                long_trans_axis[1] + frame.translation()[1],
            ],
            [
                -long_trans_axis[2] + frame.translation()[2],
                long_trans_axis[2] + frame.translation()[2],
            ],
        ]
        axises.append(axis)

    elif motion_type in [MotionType.ROT, MotionType.HELIC]:
        center_of_rotation = -jnp.cross(trans, rot)
        long_rot_axis = rot * axis_scale

        axis = [
            [frame.translation()[0], center_of_rotation[0]],
            [frame.translation()[1], center_of_rotation[1]],
            [frame.translation()[2], center_of_rotation[2]],
        ]
        axises.append(axis)

        axis = [
            [
                -long_rot_axis[0] + center_of_rotation[0],
                long_rot_axis[0] + center_of_rotation[0],
            ],
            [
                -long_rot_axis[1] + center_of_rotation[1],
                long_rot_axis[1] + center_of_rotation[1],
            ],
            [
                -long_rot_axis[2] + center_of_rotation[2],
                long_rot_axis[2] + center_of_rotation[2],
            ],
        ]
        axises.append(axis)
    return axises, motion_type


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Visualizer:
    def __init__(self: "Visualizer", ax: matplotlib.axes.Axes, title: str = ""):
        self.ax = ax
        self.ax.set_title(title)
        self.ax.set_autoscale_on(False)
        self.arrow_prop_dict = dict(
            mutation_scale=20, arrowstyle="->", shrinkA=0, shrinkB=0
        )

        # self.unit_vectors = onp.eye(3)

    def add_frame(
        self: "Visualizer",
        transformation: SE3,
        name: str = "",
        arrow_length: float = 1.0,
    ):
        # Here we create the arrows:
        base = transformation @ onp.zeros((3,))
        axis_x = transformation @ onp.array([arrow_length, 0.0, 0.0])
        axis_y = transformation @ onp.array([0.0, arrow_length, 0.0])
        axis_z = transformation @ onp.array([0.0, 0.0, arrow_length])

        a = Arrow3D(
            [base[0], axis_x[0]],
            [base[1], axis_x[1]],
            [base[2], axis_x[2]],
            **self.arrow_prop_dict,
            color="r",
        )
        self.ax.add_artist(a)
        self.ax.text(*axis_x, "x", color="r")
        a = Arrow3D(
            [base[0], axis_y[0]],
            [base[1], axis_y[1]],
            [base[2], axis_y[2]],
            **self.arrow_prop_dict,
            color="b",
        )
        self.ax.add_artist(a)
        self.ax.text(*axis_y, "y", color="b")
        a = Arrow3D(
            [base[0], axis_z[0]],
            [base[1], axis_z[1]],
            [base[2], axis_z[2]],
            **self.arrow_prop_dict,
            color="g",
        )
        self.ax.add_artist(a)
        self.ax.text(*axis_z, "z", color="g")

        self.ax.text(*base, name)

    def add_twist(
        self: "Visualizer",
        twist_: jnp.ndarray,
        frame: SE3 = SE3.identity(),
        text: str = None,
        **prop_dict,
    ):

        artists: List[Any] = []
        axises, motion_type = get_twist_axis_points(twist_, frame)

        # Hacky way to emulate matplotlibs changing color cycle
        color = prop_dict.get("color", next(self.ax._get_lines.prop_cycler)["color"])
        prop_dict["color"] = color

        for axis in axises:
            artists.extend(self.ax.plot3D(*axis, **prop_dict))

        if text in [None, ""]:
            info_text = motion_type.value
        else:
            info_text = text

        info_text_pos = frame.translation()
        self.ax.text(*info_text_pos, info_text, color=color)

        for artist in artists:
            artist.set_label(info_text)


def visualize_estimations(estimations, gts, observations, display_world=True):
    color_mapping = {
        "gt": "blue",
        "gt_sturm": "green",
        "obs_sturm": "red",
        "gt_fg_poses": "yellow",
        "obs_fg_poses": "orange",
        "gt_fg_trans": "purple",
        "obs_fg_trans": "pink",
    }

    # For 3D plotting speed we want to cap our frame amount to 20
    step_size = max(len(gts["first"]) // 20, 1)

    # Visualize
    fig = plt.figure()
    # Ground Truth
    ax1 = fig.add_subplot(121, projection="3d")
    visu = Visualizer(ax1, title="Ground Truth")
    visu.add_frame(SE3.identity(), name="World")

    world_T_est = gts["first"][0] if display_world else jaxlie.SE3.identity()

    visu.add_twist(
        estimations["gt"].twist,
        frame=world_T_est @ estimations["gt"].base_transform,
        text="gt",
        color=color_mapping["gt"],
    )
    if "gt_baseline" in estimations.keys():
        visu.add_twist(
            estimations["gt_baseline"].twist,
            frame=world_T_est @ estimations["gt_baseline"].base_transform,
            text="gt_baseline",
            color=color_mapping["gt_baseline"],
        )
    if "gt_fg" in estimations.keys():
        visu.add_twist(
            estimations["gt_fg"].twist,
            frame=world_T_est @ estimations["gt_fg"].base_transform,
            text="gt_fg",
            color=color_mapping["gt_fg"],
        )

    # visu.add_frame(gts["first"][0], name="First Body")
    if display_world:
        for i, T_world_first in enumerate(gts["first"][::step_size]):
            visu.add_frame(T_world_first, name="FB_{}".format(i))
        for i, T_world_second in enumerate(gts["second"][::step_size]):
            visu.add_frame(T_world_second, name="SB_{}".format(i))
    else:
        for i, T_first_second in enumerate(gts["first_second"][::step_size]):
            visu.add_frame(T_first_second, name="FB_SB_{}".format(i))
    ax1.legend()

    # Noisy
    ax2 = fig.add_subplot(122, projection="3d")
    visu = Visualizer(ax2, title="Noisy")
    visu.add_frame(SE3.identity(), name="World")

    for name, parameters in estimations.items():
        visu.add_twist(
            parameters.twist,
            frame=world_T_est @ parameters.base_transform,
            text=name,
            color=color_mapping[name],
        )
        visu.add_frame(
            world_T_est @ parameters.base_transform,
            name="base_transform_{}".format(name),
        )

    if display_world:
        for i, T_world_first in enumerate(observations["first"][::step_size]):
            visu.add_frame(T_world_first, name="SB_{}".format(i))
        for i, T_world_second in enumerate(observations["second"][::step_size]):
            visu.add_frame(T_world_second, name="SB_{}".format(i))
    else:
        for i, T_first_second in enumerate(observations["first_second"][::step_size]):
            visu.add_frame(T_first_second, name="FB_SB_{}".format(i))

    ax2.legend()
    return ax1, ax2


class Open3DVisualizer:
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()

    def add_twist(self, twist_, frame=SE3.identity(), name="", color=onp.zeros((3,))):
        axises_ = get_twist_axis_points(twist_, frame)[0]  # (axises_, motion_type)
        axises = onp.array(axises_)
        axises = onp.moveaxis(axises, 1, 2)
        axises = onp.reshape(axises, (axises.shape[0] * 2, 3))

        line_set = open3d.geometry.LineSet()

        lines = [[0, 1], [2, 3]]

        line_set.points = open3d.utility.Vector3dVector(axises)
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        self.vis.add_geometry(line_set)

    def add_pointcloud(self, np_pc, color=onp.zeros((3,))):
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np_pc)
        pcd.paint_uniform_color(color)
        self.vis.add_geometry(pcd)

    def add_frame(self: "Open3DVisualizer", transformation: SE3, name: str = ""):
        mesh = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        T = transformation.as_matrix()
        # T = onp.eye(4)
        # T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, onp.pi / 3, onp.pi / 2))
        # T[0, 3] = 1
        # T[1, 3] = 1.3

        mesh_t = mesh.transform(T)
        self.vis.add_geometry(mesh_t)

    def show(self):
        self.vis.run()
        self.vis.destroy_window()


class NodeShape(enum.Enum):
    factor = "s"
    variable = "o"


def visualize_graph(factors):
    G = nx.Graph()
    labels = {}
    subset_count = {}
    for factor in factors:
        # First add factor
        factor_name = type(factor).__name__
        G.add_node(factor, subset=factor_name, shape=NodeShape.factor)
        subset_count[factor_name] = subset_count.get(factor_name, 0) + 1
        labels[factor] = type(factor).__name__

        # Then loop through variables
        for variable in factor.variables:
            variable_name = type(variable).__name__
            G.add_node(
                variable,
                subset=variable_name,
                shape=NodeShape.variable,
            )
            subset_count[variable_name] = subset_count.get(factor_name, 0) + 1
            labels[variable] = variable_name

            G.add_edge(factor, variable)

    sorted_keys = sorted(subset_count)

    for node_id, node_data in G.nodes.data():
        subset_str_key = node_data["subset"]
        subset_int_key = sorted_keys.index(subset_str_key)
        G.nodes[node_id]["subset"] = subset_int_key

    fig = plt.figure(figsize=(50, 16))

    # pos = nx.spring_layout(G)
    # pos = nx.shell_layout(G)
    # pos = nx.planar_layout(G)
    pos = nx.multipartite_layout(G, align="horizontal", scale=3.0)

    nx.draw_networkx_labels(G, pos, labels=labels)
    nx.draw_networkx_edges(G, pos)

    for node_shape in NodeShape:
        node_list = [node for node in G.nodes() if G.nodes[node]["shape"] == node_shape]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_shape=node_shape.value)

    return fig
