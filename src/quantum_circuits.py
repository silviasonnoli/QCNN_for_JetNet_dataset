from math import ceil

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


__all__ = [
    "Rzyz_operator",
    "Rzxz_operator",
    "tpe_encoding_circuit",
    "hee_layer",
    "hee_n_encoding_circuit",
    "che_encoding_circuit",
    "SO4_conv_circuit",
    "SU4_conv_circuit",
    "conv_layer",
    "pool_circuit",
    "pool_layer",
    "create_2by2_qcnn_circuit_factory",
    "create_n_qubit_qcnn_circuit_factory"
]

### -- ROTATIONAL CIRCUITS -- ###

def Rzyz_operator(a, b, c):
    "Implementing the Rz(c)Ry(b)Rz(a) 1-qubit operator"
    qc = QuantumCircuit(1, name="Rz(c)Ry(b)Rz(a)")
    qc.rz(a, 0)
    qc.ry(b, 0)
    qc.rz(c, 0)
    return qc

def Rzxz_operator(a, b, c):
    "Implementing the Rz(c)Rx(b)Rz(a) 1-qubit operator"
    qc = QuantumCircuit(1, name="Rz(c)Rx(b)Rz(a)")
    qc.rz(a, 0)
    qc.rx(b, 0)
    qc.rz(c, 0)
    return qc


### -- ENCODING LAYERS -- ###

def tpe_encoding_circuit(num_qubits, x):
    qc = QuantumCircuit(num_qubits, name="TPE Encoding Layer")
    for i in range(num_qubits):
        qc.rx(x[i], i)
    return qc

def hee_layer(num_qubits, x):
    qc = QuantumCircuit(num_qubits, name="HEE Base Layer")
    qc.compose(tpe_encoding_circuit(num_qubits, x), list(range(num_qubits)), inplace=True)
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    return qc

def hee_n_encoding_circuit(num_qubits, n, x):
    if n > 0:
        qc = QuantumCircuit(num_qubits, name=f"HEE{n} Base Layer")
        for i in range(n):
            qc.compose(hee_layer(num_qubits, x), list(range(num_qubits)), inplace=True)
            qc.barrier()
        return qc
    raise ValueError("ValueError: HEE-type encoding requires at least 1 layer.\n")

def che_encoding_circuit(num_qubits, x):
    qc = QuantumCircuit(num_qubits, name="CHE Encoding Layer")
    for i in range(num_qubits):
        qc.h(i)
    for i in range(num_qubits):
        qc.rz(x[i], i)
    for i in range(num_qubits-1):
        for j in range(i+1, num_qubits):
            qc.rzz(x[i]*x[j], i, j)
    return qc


### -- CONVOLUTIONAL CIRCUITS -- ###

def SO4_conv_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(np.pi/4, 0)
    qc.rz(np.pi/4, 1)
    qc.ry(np.pi/2, 1)
    qc.cx(1, 0)
    qc.compose(Rzyz_operator(*params[:3]), [0], inplace=True)
    qc.compose(Rzyz_operator(*params[3:6]), [1], inplace=True)
    qc.cx(1, 0)
    qc.rz(-np.pi/4, 0)
    qc.ry(-np.pi/2, 1)
    qc.rz(-np.pi/4, 1)
    return qc

def SU4_conv_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(params[0], 1)
    qc.cx(0, 1)
    qc.compose(Rzxz_operator(*params[1:4]), [0], inplace=True)
    qc.compose(Rzxz_operator(*params[4:7]), [1], inplace=True)
    qc.cx(0, 1)
    qc.rx(params[7], 0)
    qc.rz(params[8], 1)
    qc.cx(0, 1)
    qc.compose(Rzxz_operator(*params[9:12]), [0], inplace=True)
    qc.compose(Rzxz_operator(*params[12:]), [1], inplace=True)
    return qc


### -- POOLING CIRCUIT -- ###

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.compose(Rzyz_operator(*params[:3]), 0, inplace=True)
    qc.compose(Rzyz_operator(*params[3:6]), 1, inplace=True)
    qc.cx(0, 1)
    qc.compose(Rzyz_operator(*params[6:]), 1, inplace=True)
    return qc


### -- LAYERS -- ###

def conv_layer(num_qubits, conv_type, params):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    n_params_SO4 = 6
    n_params_SU4 = 15
    i = 0
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        if conv_type == "SO4":
            qc = qc.compose(SO4_conv_circuit(params[i*n_params_SO4:(i+1)*n_params_SO4]), [q1, q2])
        elif conv_type == "SU4":
            qc = qc.compose(SU4_conv_circuit(params[i*n_params_SU4:(i+1)*n_params_SU4]), [q1, q2])
        else:
            raise NotImplementedError("NotImplementedError: invalid convolutional filter.\n")
        i += 1
        qc.barrier()
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        if conv_type == "SO4":
            qc = qc.compose(SO4_conv_circuit(params[i*n_params_SO4:(i+1)*n_params_SO4]), [q1, q2])
        elif conv_type == "SU4":
            qc = qc.compose(SU4_conv_circuit(params[i*n_params_SU4:(i+1)*n_params_SU4]), [q1, q2])
        i += 1
        qc.barrier()

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(sources, sinks, params):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    n_params_pool = 9
    i = 0
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[i*n_params_pool:(i+1)*n_params_pool]), [source, sink])
        i += 1
        qc.barrier()

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


### -- QCNN CIRCUITS -- ###

def create_2by2_qcnn_circuit_factory(encoding_type, conv_type):
    """
    Wrapper function to build the 4-qubit qcnn circuit.

    args:
     - encoding_type: (str): encoding type ("TPE", "HEE", "HEE2", "CHE");
     - conv_type (str): convolutional filter ("SO4", "SU4").

    returns:
     - qnn_circuit_function: function to be fed into EstimatorQNN;
     - num_input_features: #input features for EstimatorQNN;
     - num_weight_params: total number of weights;
     - input_params_vector: dummy ParameterVector for inputs;
     - weight_params_vector: dummy ParameterVector for weights.
    """
    n_qubits_total = 4
    if encoding_type not in ["TPE", "HEE", "HEE2", "CHE"]:
        raise NotImplementedError("NotImplementedError: invalid encoding type.\n")
    if conv_type not in ["SO4", "SU4"]:
        raise NotImplementedError("NotImplementedError: invalid convolutional filter.\n")

    num_input_features = n_qubits_total

    # Define base parameters for each layer type to pre-calculate total weights
    params_per_conv_circ = 6 if conv_type == "SO4" else 15
    params_per_pool_circ = 9

    # Calculate total weights needed for all layers
    total_num_weights = 4 * params_per_conv_circ + 3 * params_per_pool_circ

    # Create ParameterVectors for inputs and weights
    input_params_vector = ParameterVector("x", num_input_features)
    weight_params_vector = ParameterVector("θ", total_num_weights)

    # This function builds the circuit for EstimatorQNN.
    # It takes input_data and weights as separate arguments.
    def qnn_circuit_builder_func(input_data, weights):
        qcnn_circuit = QuantumCircuit(num_input_features, name="QCNN_Circuit")

        # Encoding Layer
        if encoding_type == "TPE":
            encoding_qc = tpe_encoding_circuit(num_input_features, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(num_input_features)), inplace=True)
        elif encoding_type == "HEE":
            encoding_qc = hee_n_encoding_circuit(num_input_features, 1, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(num_input_features)), inplace=True)
        elif encoding_type == "HEE2":
            encoding_qc = hee_n_encoding_circuit(num_input_features, 2, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(num_input_features)), inplace=True)
        elif encoding_type == "CHE":
            encoding_qc = che_encoding_circuit(num_input_features, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(num_input_features)), inplace=True)
        qcnn_circuit.barrier()

        if conv_type == "SO4":
            conv_filter = SO4_conv_circuit
        else:
            conv_filter = SU4_conv_circuit


        conv_params = weights[:params_per_conv_circ]
        current_weight_idx = params_per_conv_circ
        qcnn_circuit.compose(conv_filter(conv_params), [0, 1], inplace=True)
        conv_params = weights[current_weight_idx : current_weight_idx + params_per_conv_circ]
        current_weight_idx += params_per_conv_circ
        qcnn_circuit.compose(conv_filter(conv_params), [2, 3], inplace=True)
        conv_params = weights[current_weight_idx : current_weight_idx + params_per_conv_circ]
        current_weight_idx += params_per_conv_circ
        qcnn_circuit.compose(conv_filter(conv_params), [1, 2], inplace=True)
        qcnn_circuit.barrier()

        pool_params = weights[current_weight_idx : current_weight_idx + params_per_pool_circ]
        current_weight_idx += params_per_pool_circ
        qcnn_circuit.compose(pool_layer([0], [1], pool_params), [0, 2], inplace=True)
        pool_params = weights[current_weight_idx : current_weight_idx + params_per_pool_circ]
        current_weight_idx += params_per_pool_circ
        qcnn_circuit.compose(pool_layer([0], [1], pool_params), [1, 3], inplace=True)
        qcnn_circuit.barrier()

        conv_params = weights[current_weight_idx : current_weight_idx + params_per_conv_circ]
        current_weight_idx += params_per_conv_circ
        qcnn_circuit.compose(conv_filter(conv_params), [2, 3], inplace=True)
        qcnn_circuit.barrier()
        pool_params = weights[current_weight_idx : current_weight_idx + params_per_pool_circ]
        current_weight_idx += params_per_pool_circ
        qcnn_circuit.compose(pool_layer([0], [1], pool_params), [2, 3], inplace=True)
        qcnn_circuit.barrier()

        return qcnn_circuit

    circuit_instance = qnn_circuit_builder_func(input_params_vector, weight_params_vector)

    return circuit_instance, num_input_features, total_num_weights, input_params_vector, weight_params_vector


def create_n_qubit_qcnn_circuit_factory(n_qubits_total, encoding_type, conv_type):
    """
    Wrapper function to build the n-qubit qcnn circuit.

    args:
     - n_qubits_total: number of qubits corresponding to the input features
       (ancillary qubit is excluded);
     - encoding_type: (str): encoding type ("TPE", "HEE", "HEE2", "CHE");
     - conv_type (str): convolutional filter ("SO4", "SU4").

    returns:
     - qnn_circuit_function: function to be fed into EstimatorQNN;
     - num_input_features: #input features for EstimatorQNN;
     - num_weight_params: total number of weights;
     - input_params_vector: dummy ParameterVector for inputs;
     - weight_params_vector: dummy ParameterVector for weights.
    """
    if encoding_type not in ["TPE", "HEE", "HEE2", "CHE"]:
        raise NotImplementedError("NotImplementedError: invalid encoding type.\n")
    if conv_type not in ["SO4", "SU4"]:
        raise NotImplementedError("NotImplementedError: invalid convolutional filter.\n")

    num_input_features = n_qubits_total

    # Define base parameters for each layer type to pre-calculate total weights
    params_per_conv_circ = 6 if conv_type == "SO4" else 15
    params_per_pool_circ = 9

    # Calculate total weights needed for all layers
    total_num_weights = 0
    temp_n_layer_qubits = n_qubits_total
    while temp_n_layer_qubits > 1:

        total_num_weights += params_per_conv_circ * ceil(temp_n_layer_qubits / 2) * 2
        total_num_weights += params_per_pool_circ * ceil(temp_n_layer_qubits / 2)

        temp_n_layer_qubits = ceil(temp_n_layer_qubits / 2)


    # Create ParameterVectors for inputs and weights
    input_params_vector = ParameterVector("x", num_input_features)
    weight_params_vector = ParameterVector("θ", total_num_weights)

    # This function builds the circuit for EstimatorQNN.
    # It takes input_data and weights as separate arguments.
    def qnn_circuit_builder_func(input_data, weights):
        num_physical_qubits = n_qubits_total + 1 # Include ancillary qubit
        qcnn_circuit = QuantumCircuit(num_physical_qubits, name="QCNN_Circuit")

        # Encoding Layer
        if encoding_type == "TPE":
            encoding_qc = tpe_encoding_circuit(n_qubits_total, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(n_qubits_total)), inplace=True)
        elif encoding_type == "HEE":
            encoding_qc = hee_n_encoding_circuit(n_qubits_total, 1, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(n_qubits_total)), inplace=True)
        elif encoding_type == "HEE2":
            encoding_qc = hee_n_encoding_circuit(n_qubits_total, 2, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(n_qubits_total)), inplace=True)
        elif encoding_type == "CHE":
            encoding_qc = che_encoding_circuit(n_qubits_total, input_data)
            qcnn_circuit.compose(encoding_qc, list(range(n_qubits_total)), inplace=True)

        # QCNN Layers
        stride = 1
        active_qubits = list(range(n_qubits_total)) # Initially all data qubits are active
        current_weight_idx = 0

        while len(active_qubits) > 1:
            n_layer_qubits = len(active_qubits)

            if n_layer_qubits%2 == 0:
                conv_params = weights[current_weight_idx : current_weight_idx + params_per_conv_circ * ceil(n_layer_qubits / 2) * 2]
                current_weight_idx += params_per_conv_circ * ceil(n_layer_qubits / 2) * 2
                qcnn_circuit.compose(conv_layer(n_layer_qubits, conv_type, conv_params), active_qubits, inplace=True)
                sources = list(range(n_layer_qubits))[1::2]
                sinks = list(range(n_layer_qubits))[::2]
                pool_params = weights[current_weight_idx : current_weight_idx + params_per_pool_circ * ceil(n_layer_qubits / 2)]
                current_weight_idx += params_per_pool_circ * ceil(n_layer_qubits / 2)
                qcnn_circuit.compose(pool_layer(sources, sinks, pool_params), active_qubits, inplace=True)
            else:
                conv_params = weights[current_weight_idx : current_weight_idx + params_per_conv_circ * ceil(n_layer_qubits / 2) * 2]
                current_weight_idx += params_per_conv_circ * ceil(n_layer_qubits / 2) * 2
                qcnn_circuit.compose(conv_layer(n_layer_qubits + 1, conv_type, conv_params), active_qubits + [n_qubits_total], inplace=True)
                sources = list(range(n_layer_qubits + 1))[1::2]
                sinks = list(range(n_layer_qubits + 1))[::2]
                pool_params = weights[current_weight_idx : current_weight_idx + params_per_pool_circ * ceil(n_layer_qubits / 2)]
                current_weight_idx += params_per_pool_circ * ceil(n_layer_qubits / 2)
                qcnn_circuit.compose(pool_layer(sources, sinks, pool_params), active_qubits + [n_qubits_total], inplace=True)
            n_layer_qubits = ceil(n_layer_qubits/2)
            stride = stride*2
            active_qubits = list(range(n_qubits_total))[::stride]

        # The final observable should be on the last remaining qubit, which should be active_qubits[0]

        return qcnn_circuit

    circuit_instance = qnn_circuit_builder_func(input_params_vector, weight_params_vector)

    return circuit_instance, num_input_features, total_num_weights, input_params_vector, weight_params_vector

