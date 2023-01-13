import copy
import functools
import math
from collections import Counter
from typing import Dict, List

import error

# from typing import Set
import numpy as onp
import seaborn as sns

import helpers


class defaultkeydict(dict):
    def __missing__(self, key):
        return key


class Evaluation:
    def __init__(self):
        self.reset()

    def reset(self):
        self.errors = {}

    def append_to_error_dict(self, name, errors):
        # Ensure we are passing an array
        errors = onp.array(errors)
        error_dim = errors.shape[-1]
        self.errors[name] = onp.vstack(
            (
                self.errors.get(name, onp.empty((0, error_dim))),
                errors,
            )
        )

    def evaluate_linear_metrics(
        self,
        name,
        gt_parameters,
        pred_parameters,
        poses,
    ):
        # Based on samples we observed
        angle_error = error.translation_direction_metric(
            gt_parameters.base_transform,
            gt_parameters.twist,
            pred_parameters.base_transform,
            pred_parameters.twist,
            poses,
        )
        name_trans = name + "_translation"
        self.errors[name_trans] = onp.vstack(
            (
                self.errors.get(name_trans, onp.empty((0, 1))),
                onp.array([angle_error]),
            )
        )

        # Sampled version of full analytical twist
        similarity, angle_error = error.twist_metrics(
            gt_parameters.base_transform,
            gt_parameters.twist,
            pred_parameters.base_transform,
            pred_parameters.twist,
            poses,
        )
        name_twist = name + "_twist"
        self.errors[name_twist] = onp.vstack(
            (
                self.errors.get(name_twist, onp.empty((0, 2))),
                onp.array([similarity, angle_error]),
            )
        )
        return {
            "translation": onp.array([angle_error]),
            "twist": onp.array([similarity, angle_error]),
        }

    def get_error_dict(self):
        return {
            "errors": self.errors,
        }

    @staticmethod
    def from_error_dict(error_dict, ignore_gt=False):
        eval = Evaluation()
        eval.errors = error_dict["errors"]
        eval.errors = {
            name: val
            for name, val in eval.errors.items()
            if not ignore_gt or (ignore_gt and not "gt" in name)
        }

        return eval

    def add_error_dict(self, error_dict, rename: Dict[str, str] = helpers.KeyDict):
        self.errors.update(helpers.extract_dict(error_dict["errors"], rename))
        self.correct_joints_type.update(
            helpers.extract_dict(error_dict["correct_joints"], rename)
        )
