from typing import Literal, Optional, TypedDict
import torch
import pennylane as qml
import pennylane.templates as qmlt
import pennylane.qnn.torch as qml_qnn_torch
import math
import numpy as np

from transformer.pytorch.quantum.pennylane.fable import FABLE


class PennyLaneArgs(TypedDict):
    hadamard: Optional[bool]
    encoder: Literal["angle", "amplitude"]
    angle_rot: Literal["X", "Y", "Z"]
    entangler: Literal["basic", "strong"]
    rot: Optional[Literal["X", "Y", "Z"]]
    imprimitive: Optional[Literal["X", "Y", "Z"]]


class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        n_qubits,
        n_qlayers=1,
        q_device="default.qubit",
        pennylane_args: PennyLaneArgs = {
            "hadamard": False,
            "encoder": "angle",
            "angle_rot": "Z",
            "entangler": "basic",
            "rot": "Z",
            "imprimitive": "Z",
        },
        embed_dim=4,
    ):
        super().__init__()

        assert (
            math.log2(embed_dim) % 2 == 0
        ), "Embedding dimension should fit 2**n * 2**n."

        assert (
            n_qubits == int(np.log2(embed_dim)) // 2
        ), "Number of qubits should be equal to matrix_dim"

        hadamard = pennylane_args.get("hadamard", False)
        encoder = pennylane_args.get("encoder", "angle")
        angle_rot = pennylane_args.get("angle_rot", "Z")
        rot = pennylane_args.get("rot", "Z")
        entangler = pennylane_args.get("entangler", "basic")
        imprimitive = pennylane_args.get("imprimitive", "CNOT")

        ancilla_wires = ["ancilla"]

        matrix_dim = int(np.log2(embed_dim)) // 2
        wires_i = [f"i{index}" for index in range(matrix_dim)]
        wires_j = [f"j{index}" for index in range(matrix_dim)]

        self.matrix_dim = matrix_dim

        if q_device == "default.qubit.torch":
            dev = qml.device(
                q_device, wires=ancilla_wires + wires_i + wires_j, torch_device="cuda"
            )
        else:
            dev = qml.device(q_device, wires=ancilla_wires + wires_i + wires_j)

        if rot == "X":
            rot = qml.RX  # type: ignore
        elif rot == "Y":
            rot = qml.RY  # type: ignore
        else:
            rot = qml.RZ  # type: ignore

        if imprimitive == "X":
            imprimitive = qml.CNOT
        elif imprimitive == "Y":
            imprimitive = qml.CY
        else:
            imprimitive = qml.CZ

        @qml.qnode(dev, interface="torch")
        def qlayer(inputs, weights):
            inputs = inputs.reshape(-1, 2**self.matrix_dim, 2**self.matrix_dim)
            inputs = torch.vmap(lambda x: x / torch.max(torch.abs(x)))(inputs)
            FABLE(inputs, wires=ancilla_wires + wires_i + wires_j, tol=0)
            if entangler == "strong":
                qmlt.StronglyEntanglingLayers(
                    weights, wires=ancilla_wires + wires_i, imprimitive=imprimitive
                )
            else:
                qmlt.BasicEntanglerLayers(
                    weights, wires=ancilla_wires + wires_i, rotation=rot
                )
            return [qml.expval(qml.PauliZ(wires=i)) for i in ancilla_wires + wires_i]

        weight_shapes = (
            qml.StronglyEntanglingLayers.shape(
                n_layers=n_qlayers, n_wires=len(ancilla_wires + wires_i)
            )
            if entangler == "strong"
            else (n_qlayers, len(ancilla_wires + wires_i))
        )

        self.linear = qml_qnn_torch.TorchLayer(
            qlayer,
            {"weights": weight_shapes},
            init_method=torch.nn.init.kaiming_normal_,
        )

        self.squeeze = torch.nn.Linear(
            len(ancilla_wires + wires_i), len(ancilla_wires + wires_i) - 1
        )

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.squeeze(x)
        return x
