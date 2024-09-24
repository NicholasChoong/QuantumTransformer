import torch
import pennylane as qml
import pennylane.templates as qmlt
import pennylane.qnn.torch as qml_qnn_torch


class QuantumLayer(torch.nn.Module):
    def __init__(self, n_qubits, n_qlayers=1, q_device="default.qubit"):
        super().__init__()

        if q_device == "default.qubit.torch":
            dev = qml.device(q_device, wires=n_qubits, torch_device="cuda")
        else:
            dev = qml.device(q_device, wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qlayer(inputs, weights):
            # qmlt.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qmlt.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
            qmlt.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RZ)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.linear = qml_qnn_torch.TorchLayer(qlayer, weight_shapes)

    def forward(self, inputs):
        return self.linear(inputs)
