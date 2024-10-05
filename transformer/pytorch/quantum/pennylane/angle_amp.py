from typing import Literal, Optional, TypedDict
import torch
import pennylane as qml
import pennylane.templates as qmlt
import pennylane.qnn.torch as qml_qnn_torch


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

        hadamard = pennylane_args.get("hadamard", False)
        encoder = pennylane_args.get("encoder", "angle")
        angle_rot = pennylane_args.get("angle_rot", "Z")
        rot = pennylane_args.get("rot", "Z")
        entangler = pennylane_args.get("entangler", "basic")
        imprimitive = pennylane_args.get("imprimitive", "CNOT")

        self.encoder = encoder

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

        if imprimitive == "X":
            imprimitive = qml.CNOT
        elif imprimitive == "Y":
            imprimitive = qml.CY
        else:
            imprimitive = qml.CZ

        @qml.qnode(dev, interface="torch")
        def qlayer(inputs, weights):
            if hadamard:
                qml.Hadamard(wires=range(n_qubits))
            if encoder == "angle":
                qmlt.AngleEmbedding(inputs, wires=range(n_qubits), rotation=angle_rot)
            else:
                qmlt.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
            if entangler == "strong":
                qmlt.StronglyEntanglingLayers(
                    weights, wires=range(n_qubits), imprimitive=imprimitive
                )
            else:
                qmlt.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=rot)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = (
            qml.StronglyEntanglingLayers.shape(n_layers=n_qlayers, n_wires=n_qubits)
            if entangler == "strong"
            else (n_qlayers, n_qubits)
        )

        # if encoder == "amplitude":
        #     qlayer = qml.transforms.broadcast_expand(qlayer)

        self.linear = qml_qnn_torch.TorchLayer(
            qlayer,
            {"weights": weight_shapes},
            init_method=torch.nn.init.kaiming_normal_,
        )

    def forward(self, inputs):
        if self.encoder == "amplitude":
            batch_size, seq_len, embed_dim = inputs.size()
            inputs = inputs.reshape(-1, embed_dim)
        return self.linear(inputs)
