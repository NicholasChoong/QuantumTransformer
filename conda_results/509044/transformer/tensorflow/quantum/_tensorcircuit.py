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


def get_circuit():
    qpred_batch = K.vmap(qpred, vectorized_argnums=0)
    qpred_batch = tc.backend.jit(qpred_batch)
    return qpred_batch


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_qlayers=1, **_):
        super(QuantumLayer, self).__init__()

        # Parameter initialization similar to PyTorch's kaiming_normal_
        self.quantum_weights = self.add_weight(
            shape=(n_qlayers, n_qubits),
            initializer=tf.keras.initializers.HeNormal(),  # Equivalent to kaiming_normal_
            trainable=True,
        )

        # Get the quantum circuit with TensorFlow interface
        self.linear = get_circuit()

    def call(self, inputs):
        shape = tf.shape(inputs)
        # Flatten input to prepare for quantum circuit
        inputs = tf.reshape(inputs, [-1, shape[-1]])

        # Pass inputs and weights to the quantum circuit
        outputs = self.linear(inputs, self.quantum_weights)

        # Reshape outputs to the original shape
        outputs = tf.reshape(outputs, shape)
        return outputs
