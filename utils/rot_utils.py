# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import Union

import numpy as np
import pytorch3d.transforms as pt

import torch
from scipy.spatial.transform import Rotation as R


def align_quat_to_styleA(q_xyzw: np.ndarray, hand: str) -> np.ndarray:
    S_LEFT = np.array(
        [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ]
    )
    S_RIGHT = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    Rm = R.from_quat(q_xyzw).as_matrix()
    S = S_LEFT if hand.lower().startswith("l") else S_RIGHT

    R_aligned = Rm @ S

    fix_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)

    R_final = R_aligned @ fix_matrix
    return R.from_matrix(R_final).as_quat()


class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [
                getattr(pt, f"{from_rep}_to_matrix"),
                getattr(pt, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(func, convention=from_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                getattr(pt, f"matrix_to_{to_rep}"),
                getattr(pt, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [
                    functools.partial(func, convention=to_convention) for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: Union[np.ndarray, torch.Tensor], funcs: list
    ) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


quat_to_rot6d = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")
