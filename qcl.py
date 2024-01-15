import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
from numpy.random import default_rng

import collections

import matplotlib.pyplot as plt
#from cirq.contrib.svg import SVGCircuit

def convert_to_circuit(x):
    """Encode truncated classical image into quantum datapoint."""
    y = np.arcsin(x)
    z = np.arccos(x**2)
    qubits = cirq.GridQubit.rect(5, 1)
    circuit = cirq.Circuit()
    for i in range(5):
        circuit.append(cirq.ry(y).on(qubits[i]))
        circuit.append(cirq.rz(z).on(qubits[i]))
    return circuit

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout
    
    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)
            
    def add_layer_single(self,circuit,gate,prefix):
        symbol = sympy.Symbol(prefix + '-' + str(0))
        circuit.append(gate(symbol).on(self.readout))
    
    def add_entangler(self,circuit,len_qubit):
        circuit.append(cirq.CZ(self.readout,self.data_qubits[0]))
        for i in range(len_qubit-1):
            circuit.append(cirq.CZ(self.data_qubits[i],self.data_qubits[(i+1)%len_qubit]))
        circuit.append(cirq.CZ(self.readout,self.data_qubits[-1]))

def create_quantum_model(c_depth=3):
    data_qubits = cirq.GridQubit.rect(5,1)
    readout = cirq.GridQubit(-1,-1)
    circuit = cirq.Circuit()
    
    circuit.append(cirq.H(readout))
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout = readout
    )
    
    for i in range(3):
        builder.add_entangler(circuit,5)
        builder.add_layer(circuit, gate = cirq.XX, prefix='xx'+str(i))
        builder.add_layer(circuit, gate = cirq.ZZ, prefix='zz'+str(i))
        builder.add_layer(circuit, gate = cirq.XX, prefix='xx1'+str(i))
        builder.add_layer_single(circuit, gate = cirq.rz, prefix='z1'+str(i))
        builder.add_layer_single(circuit, gate = cirq.rx, prefix='x1'+str(i))
        builder.add_layer_single(circuit, gate = cirq.rz, prefix='z2'+str(i))
        
    
    return circuit, cirq.Z(readout)

def generate_noisy_sine(x_min, x_max, num_x):
    rng = default_rng(0)
    x_train = [[rng.uniform(x_min, x_max)] for _ in range(num_x)]
    y_train = [np.sin(np.pi * x[0]) for x in x_train]
    mag_noise = 0.01
    y_train += mag_noise * rng.random(num_x)
    # return np.array(x_train), np.array(y_train)
    return np.array(x_train).flatten(), np.array(y_train)


x_min = -1.0
x_max = 1.0
num_x = 80
x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)
x_test, y_test = generate_noisy_sine(x_min, x_max, num_x)

model_circuit, model_readout = create_quantum_model()

# Build the Keras model.
model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['mae'])

print(model.summary())

x_train_circ = [convert_to_circuit(x) for x in x_train]
x_test_circ = [convert_to_circuit(x) for x in x_test]
x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)


EPOCHS = 100
BATCH_SIZE = 50

qnn_history = model.fit(
      x_train_tfcirc, y_train,
      batch_size=25,
      epochs=EPOCHS,
      verbose=1,
      validation_data=(x_test_tfcirc,y_test)
)

model.evaluate(x_test_tfcirc,y_test)
y_pred = model.predict(x_test_tfcirc)

plt.plot(x_test,y_pred,"o",label="pred")
plt.plot(x_test,y_test,"xr",label="test")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
#plt.show()
plt.savefig('fig-depth5.png')
