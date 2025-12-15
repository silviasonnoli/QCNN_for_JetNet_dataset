import os
import numpy as np
from tqdm import tqdm
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

from .quantum_circuits import create_2by2_qcnn_circuit_factory, create_n_qubit_qcnn_circuit_factory

### -- LOSS FUNCTIONS AND GRADIENTS -- ###

def cross_entropy_loss(predictions, true_labels):
    predictions = predictions.reshape(-1)
    probs = (predictions + 1) / 2
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    true_labels_01 = (true_labels + 1) / 2
    loss = -np.mean(true_labels_01 * np.log(probs) + (1 - true_labels_01) * np.log(1 - probs))
    return loss

def cross_entropy_grad(predictions, true_labels):
    predictions = predictions.reshape(-1)
    # map true_labels and predictions to [0, 1]
    true_labels_01 = (true_labels + 1) / 2
    probs = (predictions + 1) / 2

    d_loss_d_probs = - (true_labels_01 / (probs + 1e-10) - (1 - true_labels_01) / (1 - probs + 1e-10))

    d_loss_d_output = d_loss_d_probs / 2
    return d_loss_d_output

def mse_loss(predictions, true_labels):
    return np.mean((predictions.reshape(-1) - true_labels)**2)

def mse_grad(predictions, true_labels):
    return 2 * (predictions.reshape(-1) - true_labels)

class QCNN_training():
    def __init__(self, N_components, encoding_type, conv_type, loss="bce", pretrained_weights=None):
        """
        Container function for QCNN training. Sets up environment for
        training and initialises EstimatorQNN.
        args:
         - N_components: number of qubits relative to the data dimensions (excluding ancilla);
         - encoding_type: type of encoding circuit;
         - conv_type: typer of convolutional circuit;
         - loss: loss to use (choose between "bce" and "mse";
         - pretrained_weights: if present, initialises weights with provided pretrained ones.
        """
        if loss=="bce":
            self.calculate_loss = cross_entropy_loss
            self.grad_loss = cross_entropy_grad
        elif loss=="mse":
            self.calculate_loss = mse_loss
            self.grad_loss = mse_grad
        else:
            raise ValueError("ValueError: invalid loss type")

        self.N_components = N_components
        if N_components == 4:
            circuit, num_input_features, self.num_weight_params, input_params_vector, weight_params_vector = \
                create_2by2_qcnn_circuit_factory(encoding_type, conv_type)
        else:
            circuit, num_input_features, self.num_weight_params, input_params_vector, weight_params_vector = \
                create_n_qubit_qcnn_circuit_factory(N_components, encoding_type, conv_type)

        if N_components == 4:
            observable = SparsePauliOp("Z" + (N_components - 1)*"I")
        else:
            observable = SparsePauliOp(N_components*"I" + "Z")
        estimator = Estimator()

        self.current_weights = pretrained_weights
        if self.current_weights is not None:
            if len(self.current_weights) != self.num_weight_params:
                raise ValueError(f"ValueError: pretrained weights size ({len(self.current_weights)}) "
                                 f"incompatible with number of parameters ({self.num_weight_params}).")

        self.qnn = EstimatorQNN(
            circuit=circuit,
            input_params=input_params_vector,
            weight_params=weight_params_vector,
            observables=[observable],
            estimator=estimator
        )

        # Dummy circuit
        x_data = np.random.rand(num_input_features)
        initial_weights = np.random.normal(0, 0.01, self.num_weight_params)

        output = self.qnn.forward(input_data=x_data, weights=initial_weights)
        print(f"QCNN utput: {output}")

        gradients = self.qnn.backward(input_data=x_data, weights=initial_weights)
        print(f"QCNN gradients: {gradients}")

        test_input_data = np.random.rand(num_input_features)
        test_weights = np.random.rand(self.num_weight_params)
        drawing_circuit = circuit.assign_parameters({**dict(zip(input_params_vector, test_input_data)),
                                                **dict(zip(weight_params_vector, test_weights))})

        print("\nExample of QCNN circuit:")
        circuit.draw("mpl", style="clifford")

    def Adam(self, X_train_full, y_train_full, X_test_full, y_test_full, save_weights=False, **kwargs):
        """
        Implementation of Adam optimisation algorithm for a qnn circuit model.

        args:
         - X_train_full: full swept of the training x-s;
         - y_train_full: full swept of the training y-s;
         - X_test_full: full swept of the test x-s;
         - y_test_full: full swept of the test y-s;
         - save_weights: flag to save weights to file;
         - **kwargs: must contain the training hyperparameters from the config file.

        returns:
         - history: dictionary containing the metrics history of training.
        """
        history = {
            "accuracy_tr" : [],
            "accuracy_ts" : [],
            "loss_tr" : [],
            "loss_ts" : []
          }
        # map labels to [-1, 1]
        y_train_full_mapped = 2 * y_train_full - 1
        y_test_full_mapped = 2 * y_test_full - 1

        num_samples_total = len(X_train_full)

        # Adam inizialisation
        m = np.zeros(self.num_weight_params)
        v = np.zeros(self.num_weight_params)
        t = 0

        # If no pretrained weights are provided, initialise model weights
        if self.current_weights is None:
            self.current_weights = np.random.normal(0, kwargs["init_weights_stdev"], self.num_weight_params)

        print(f"Launching training with Adam and mini-batching (batch_size={kwargs['batch_size']})")

        for epoch in range(kwargs["num_epochs"]):
            # Mescola i dati
            permutation = np.random.permutation(num_samples_total)
            X_shuffled = X_train_full[permutation]
            y_shuffled = y_train_full_mapped[permutation]

            epoch_loss_total = 0.0
            num_batches = 0

            for i in tqdm(range(0, num_samples_total, kwargs["batch_size"]), desc='% of epoch swept'):
                X_batch = X_shuffled[i : i + kwargs["batch_size"]]
                y_batch = y_shuffled[i : i + kwargs["batch_size"]]

                t += 1

                current_predictions = self.qnn.forward(input_data=X_batch, weights=self.current_weights)

                grad_output_vs_weights = self.qnn.backward(input_data=X_batch, weights=self.current_weights)[1]

                grad_loss_vs_output = self.grad_loss(current_predictions, y_batch)
                grad_output_vs_weights = grad_output_vs_weights.squeeze()

                final_gradients = grad_loss_vs_output.reshape(-1, 1) * grad_output_vs_weights

                avg_gradients = np.mean(final_gradients, axis=0)

                # L1 Regularisation
                L1_grad = kwargs["L1_lambda"] * np.sign(self.current_weights)

                # L1 Regularisation
                L2_grad = 2 * kwargs["L2_lambda"] * self.current_weights

                m = kwargs["beta1"] * m + (1 - kwargs["beta1"]) * avg_gradients
                v = kwargs["beta2"] * v + (1 - kwargs["beta2"]) * (avg_gradients ** 2)
                m_hat = m / (1 - kwargs["beta1"]**t)
                v_hat = v / (1 - kwargs["beta2"]**t)
                self.current_weights -= kwargs["learning_rate"] * (kwargs["decay_rate"] ** epoch) * \
                                   m_hat / (np.sqrt(v_hat) + kwargs["epsilon"]) + L1_grad + L2_grad


                current_batch_predictions = self.qnn.forward(input_data=X_batch, weights=self.current_weights)
                current_batch_loss = self.calculate_loss(current_batch_predictions, y_batch)

                epoch_loss_total += current_batch_loss
                num_batches += 1

            y_predictions_tr = self.qnn.forward(input_data=X_train_full, weights=self.current_weights)
            avg_epoch_loss = self.calculate_loss(y_predictions_tr, y_train_full)
            y_predictions_tr = (y_predictions_tr > 0).astype(int).reshape(-1)
            accuracy_tr = np.mean(y_predictions_tr == y_train_full) * 100
            test_predictions = self.qnn.forward(input_data=X_test_full, weights=self.current_weights)
            test_loss = self.calculate_loss(test_predictions, y_test_full_mapped)
            test_predictions = (test_predictions > 0).astype(int).reshape(-1)
            accuracy_ts = np.mean(test_predictions == y_test_full) * 100
            print(f"Epoca {epoch+1}/{kwargs['num_epochs']}")
            print(f"Train: average loss: {avg_epoch_loss:.4f}, accuracy: {accuracy_tr:.2f}")
            print(f"Test:  average loss: {test_loss:.4f}, accuracy: {accuracy_ts:.2f}")
            history["accuracy_tr"].append(accuracy_tr)
            history["accuracy_ts"].append(accuracy_ts)
            history["loss_tr"].append(avg_epoch_loss)
            history["loss_ts"].append(test_loss)
        return history

    def average_performance(self, X_train_full, y_train_full, X_test_full, y_test_full, n_trials=50):
        """
        Evaluates model average performance on a number of trials.

        args:
         - X_train_full: full swept of the training x-s;
         - y_train_full: full swept of the training y-s;
         - X_test_full: full swept of the test x-s;
         - y_test_full: full swept of the test y-s;
         - n_trials: number of evaluations to perform.
        """
        y_train_full_mapped = 2 * y_train_full - 1
        y_test_full_mapped = 2 * y_test_full - 1

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        for i in range(n_trials):
            train_predictions = self.qnn.forward(input_data=X_train_full, weights=self.current_weights)
            train_loss.append(self.calculate_loss(train_predictions, y_train_full_mapped))

            train_predictions = (train_predictions > 0).astype(int).reshape(-1) # Se output [-1, 1], >0 è classe 1, <=0 è classe 0
            train_accuracy.append(np.mean(train_predictions == y_train_full))

            test_predictions = self.qnn.forward(input_data=X_test_full, weights=self.current_weights)
            test_loss.append(self.calculate_loss(test_predictions, y_test_full_mapped))

            test_predictions = (test_predictions > 0).astype(int).reshape(-1) # Se output [-1, 1], >0 è classe 1, <=0 è classe 0
            test_accuracy.append(np.mean(test_predictions == y_test_full))

        avg_loss_train = np.mean(train_loss)
        avg_accuracy_train = np.mean(train_accuracy)

        std_loss_train = np.std(train_loss)
        std_accuracy_train = np.std(train_accuracy)

        avg_loss_test = np.mean(test_loss)
        avg_accuracy_test = np.mean(test_accuracy)

        std_loss_test = np.std(test_loss)
        std_accuracy_test = np.std(test_accuracy)

        print(f"Average loss on training set: {avg_loss_train:.4f} +- {std_loss_train:.4f}")
        print(f"Average accuracy on training set: {avg_accuracy_train*100:.2f} +- {std_accuracy_train*100:.2f} %")
        print(f"\nAverage loss on test set: {avg_loss_test:.4f} +- {std_loss_test:.4f}")
        print(f"Average accuracy on test set: {avg_accuracy_test*100:.2f} +- {std_accuracy_test*100:.2f} %")

    def save_weights(self):
        weights_dir = ".pretrained"
        os.makedirs(weights_dir, exist_ok=True)
        np.save(os.path.join(weights_dir, f'pretrained_{self.N_components}_qcnn_weights.npy'), self.current_weights)
