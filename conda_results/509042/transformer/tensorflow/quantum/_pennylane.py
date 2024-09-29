import tensorflow as tf
import pennylane as qml
import pennylane.templates as qmlt
from pennylane.qnn import KerasLayer


class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits, n_qlayers=1, q_device="default.qubit"):
        super(QuantumLayer, self).__init__()

        dev = qml.device(q_device, wires=n_qubits)

        @qml.qnode(dev, interface="tf")
        def qlayer(inputs, weights):
            # qmlt.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qmlt.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
            qmlt.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RZ)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        self.linear = KerasLayer(qlayer, weight_shapes, output_dim=n_qubits)
        # self.jitted_qlayer = tf.function(qlayer, jit_compile=True)
        # self.linear = qlayer

        # self.quantum_weights = self.add_weight(
        #     shape=(n_qlayers, n_qubits),
        #     initializer=tf.keras.initializers.HeNormal(),  # Equivalent to kaiming_normal_
        #     trainable=True,
        # )

    def call(self, inputs):
        shape = inputs.shape
        x = tf.reshape(inputs, [-1, shape[-1]])
        x = [self.linear(embed) for embed in x]
        # x = tf.cast(x, tf.float64)
        # print(self.linear(x[0]))
        # x = tf.map_fn(self.linear, x, fn_output_signature=tf.float64)
        # for i, embed in enumerate(x):
        #     print(i)
        #     self.linear(embed)
        x = tf.concat(x, axis=-1)
        x = tf.reshape(x, shape)
        return x
