from typing import Literal
import torch
import pennylane as qml
import pennylane.templates as qmlt
import pennylane.qnn.torch as qml_qnn_torch


class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        n_qubits,
        n_qlayers=1,
        q_device="default.qubit",
        pennylane_args={},
    ):
        super().__init__()

        hadamard = pennylane_args.get("hadamard", False)
        encoder = pennylane_args.get("encoder", "angle")
        angle_rot = pennylane_args.get("angle_rot", "Z")
        rot = pennylane_args.get("rot", "Z")
        entangler = pennylane_args.get("entangler", "basic")
        imprimitive = pennylane_args.get("imprimitive", "CNOT")

        if q_device == "default.qubit.torch":
            dev = qml.device(q_device, wires=n_qubits, torch_device="cuda")
        else:
            dev = qml.device(q_device, wires=n_qubits)

        if rot == "X":
            rot = qml.RX  # type: ignore
        elif rot == "Y":
            rot = qml.RY  # type: ignore
        else:
            rot = qml.RZ  # type: ignore

        @qml.qnode(dev, interface="torch")
        def qlayer(inputs, weights):
            if hadamard:
                qml.Hadamard(wires=range(n_qubits))
            if encoder == "angle":
                qmlt.AngleEmbedding(inputs, wires=range(n_qubits), rotation=angle_rot)
            else:
                qmlt.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            if entangler == "strong":
                qmlt.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            else:
                qmlt.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=rot)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.linear = qml_qnn_torch.TorchLayer(qlayer, weight_shapes)

    def forward(self, inputs):
        return self.linear(inputs)