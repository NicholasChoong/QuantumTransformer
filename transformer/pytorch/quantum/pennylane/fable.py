# type: ignore

import warnings

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane.wires import Wires

import torch


class FABLE(Operation):

    num_wires = AnyWires

    num_params = 1

    grad_method = None

    def __init__(self, input_matrix, wires, tol=0, id=None):
        wires = Wires(wires)
        if not qml.math.is_abstract(input_matrix):
            if qml.math.any(qml.math.iscomplex(input_matrix)):
                raise ValueError(
                    "Support for imaginary values has not been implemented."
                )

            alpha = qml.math.linalg.norm(qml.math.ravel(input_matrix), np.inf)
            if alpha > 1:
                raise ValueError(
                    "The subnormalization factor should be lower than 1."
                    + "Ensure that the values of the input matrix are within [-1, 1]."
                )
        else:
            if tol != 0:
                raise ValueError(
                    "JIT is not supported for tolerance values greater than 0. Set tol = 0 to run."
                )

        batch, row, col = qml.math.shape(input_matrix)
        if row != col:
            warnings.warn(
                f"The input matrix should be of shape NxN, got {input_matrix.shape}."
                + "Zeroes were padded automatically."
            )
            dimension = max(row, col)
            input_matrix = qml.math.pad(
                input_matrix, ((0, dimension - row), (0, dimension - col))
            )
            batch, row, col = qml.math.shape(input_matrix)
        n = int(qml.math.ceil(qml.math.log2(col)))
        if n == 0:
            n = 1
        if col < 2**n:
            input_matrix = qml.math.pad(
                input_matrix, ((0, 2**n - col), (0, 2**n - col))
            )
            col = 2**n
            warnings.warn(
                "The input matrix should be of shape NxN, where N is a power of 2."
                + f"Zeroes were padded automatically. Input is now of shape {input_matrix.shape}."
            )

        if len(wires) != 2 * n + 1:
            raise ValueError(
                f"Number of wires is incorrect, expected {2*n+1} but got {len(wires)}"
            )

        self._hyperparameters = {"tol": tol}

        super().__init__(input_matrix, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(
        inputs_matrix, wires, tol=0
    ):  # pylint:disable=arguments-differ
        op_list = []
        alphas = torch.arccos(inputs_matrix).reshape(inputs_matrix.shape[0], -1)
        thetas = torch.stack([compute_theta(alpha) for alpha in alphas])

        batched = qml.math.ndim(thetas) > 1

        thetas_batch = qml.math.T(thetas) if batched else [thetas]

        ancilla = [wires[0]]
        wires_i = wires[1 : 1 + len(wires) // 2][::-1]
        wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]

        code = gray_code((2 * qml.math.log2(len(inputs_matrix[0]))))
        n_selections = len(code)

        control_wires = [
            int(qml.math.log2(int(code[i], 2) ^ int(code[(i + 1) % n_selections], 2)))
            for i in range(n_selections)
        ]
        wire_map = dict(enumerate(wires_j + wires_i))
        for w in wires_i:
            op_list.append(qml.Hadamard(w))

        # for thetas in thetas_batch:
        for i, (thetas, control_index) in enumerate(zip(thetas_batch, control_wires)):
            op_list.append(qml.RZ(2 * thetas, wires=ancilla))
            op_list.append(qml.CNOT(wires=[wire_map[control_index]] + ancilla))

        for w_i, w_j in zip(wires_i, wires_j):
            op_list.append(qml.SWAP(wires=[w_i, w_j]))

        for w in wires_i:
            op_list.append(qml.Hadamard(w))

        return op_list
