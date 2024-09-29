import torch

import tensorcircuit as tc
import tensorflow as tf

K = tc.set_backend("tensorflow")


def angle_embedding(c: tc.Circuit, inputs):
    num_qubits = inputs.shape[-1]

    for j in range(num_qubits):
        c.rx(j, theta=inputs[j])  # type: ignore


def basic_vqc(c: tc.Circuit, inputs, weights):
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    for i in range(num_qlayers):
        for j in range(num_qubits):
            c.rx(j, theta=weights[i, j])  # type: ignore
        if num_qubits == 2:
            c.cnot(0, 1)  # type: ignore
        elif num_qubits > 2:
            for j in range(num_qubits):
                c.cnot(j, (j + 1) % num_qubits)  # type: ignore


def qpred(inputs, weights):
    """
    Equivalent to the following PennyLane circuit:
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
    """

    num_qubits = inputs.shape[-1]

    c = tc.Circuit(num_qubits)
    angle_embedding(c, inputs)
    basic_vqc(c, inputs, weights)

    ypred = tf.convert_to_tensor(
        [c.expectation_ps(z=[i]) for i in range(weights.shape[1])]
    )
    ypred = K.real(ypred)
    return ypred


def get_circuit(
    torch_interface: bool = False,
):
    qpred_batch = K.vmap(qpred, vectorized_argnums=0)
    if torch_interface:
        qpred_batch = tc.interfaces.torch_interface(qpred_batch, jit=True)

    return qpred_batch


class QuantumLayer(torch.nn.Module):
    def __init__(self, num_qubits, num_qlayers=1, **_):
        super().__init__()

        self.weights = torch.nn.Parameter(
            torch.nn.init.kaiming_normal_(torch.empty(num_qlayers, num_qubits))
        )
        self.linear = get_circuit(torch_interface=True)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = self.linear(x, self.weights)
        x = x.reshape(shape)
        return x
