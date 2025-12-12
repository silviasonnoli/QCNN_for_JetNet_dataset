import os
import json
import argparse
from IPython.display import clear_output
import numpy as np
import torch
from qiskit_machine_learning.utils import algorithm_globals

from src.preprocessing import download_toptagging_dataset, preprocessing
from src.training import QCNN_training
from src.CNN import CNN_training
from src.utils import print_training_history

np.random.seed(1)
algorithm_globals.random_seed = 12345

def parsing():
    parser = argparse.ArgumentParser(description='Run QCNN simulation')
    parser.add_argument('--config_file', metavar='config_file', dest='config_file',
            help='Name of the config file of choice for the model hyperparameters.')
    parser.add_argument('--train_size', metavar='train_size', dest='train_size', type=int,
            help='Size of training set.')
    parser.add_argument('--test_size', metavar='test_size', dest='test_size', type=int,
            help='Size of test set.')
    parser.add_argument('--N_components', metavar='N_components', dest='N_components', type=int,
            help='#components for the PCA image, and thus #qubits of the QCNN.')
    parser.add_argument('--model_type', metavar='model_type', dest='model_type',
            help='Whether to train quantum (\"qcnn\", default) or classical (\"cnn\") model.')
    parser.add_argument('--encoding_type', metavar='encoding_type', dest='encoding_type',
            help='Type of the encoding circuit for the QCNN. Default to  \"TPE\".')
    parser.add_argument('--conv_type', metavar='conv_type', dest='conv_type',
            help='Type of convolutional circuit used in the QCNN. Default to \"SO4\".')
    parser.add_argument('--loss_type', metavar='loss_type', dest='loss_type',
            help='Type of loss to use during training. Choose between \"bce\" (default) and \"mse\".')
    parser.add_argument('--pretrained_weights_file', metavar='pretrained_weights_file', dest='pretrained_weights_file',
            help='Name of the pretrained weights file if any. Default to None')
    parser.add_argument('--save_weights', metavar='save_weights', dest='save_weights', type=bool,
            help='Whether to save or not the execution weigths. Deafult to False')
    parser.set_defaults(N_components=4)
    parser.set_defaults(train_size=160)
    parser.set_defaults(test_size=40)
    parser.set_defaults(model_type="qcnn")
    parser.set_defaults(encoding_type="TPE")
    parser.set_defaults(conv_type="SO4")
    parser.set_defaults(loss_type="bce")
    parser.set_defaults(pretrained_weights_file=None)
    parser.set_defaults(save_weights=False)
    args = parser.parse_args()
    return args

def main():
    args = parsing()
    N_components = args.N_components
    train_size = args.train_size
    test_size = args.test_size
    model_type = args.model_type
    encoding_type = args.encoding_type
    conv_type = args.conv_type
    loss_type = args.loss_type
    pretrained_weights_file = args.pretrained_weights_file
    save_weights = args.save_weights
    if pretrained_weights_file:
        if model_type == "qcnn":
            pretrained_weights = np.load(os.path.join('.pretrained', pretrained_weights_file))
        elif model_type == "cnn":
            pretrained_weights = torch.load(os.path.join('.pretrained', pretrained_weights_file))
    else:
        pretrained_weights=None
    with open(os.path.join('configs', args.config_file), 'r') as f:
        hyp_config = json.load(f)

    particle_train, jet_train, particle_test, jet_test = download_toptagging_dataset(train_size, test_size)

    jet_train_pca, jet_y_train, jet_test_pca, jet_y_test = preprocessing(
                                                                particle_train,
                                                                particle_test,
                                                                jet_train,
                                                                jet_test,
                                                                N_components
                                                                )

    if model_type == "qcnn":
        QCNN_train = QCNN_training(N_components, encoding_type, conv_type, loss_type, pretrained_weights)
        qcnn_history = QCNN_train.Adam(jet_train_pca, jet_y_train, jet_test_pca, jet_y_test, save_weights, **hyp_config)
        print_training_history(qcnn_history,
                               model_type=model_type,
                               N_components=N_components,
                               encoding_type=encoding_type,
                               conv_type=conv_type,
                               loss_type=loss_type)
        QCNN_train.average_performance(jet_train_pca, jet_y_train, jet_test_pca, jet_y_test, n_trials=50)
        if save_weights:
            QCNN_train.save_weights()
    elif model_type == "cnn":
        out_channels = 10 if conv_type=="SO4" else 17
        kernel_size = 2
        CNN_train = CNN_training(N_components, out_channels, kernel_size, 'cpu', pretrained_weights, **hyp_config)
        cnn_history = CNN_train.train(jet_train_pca, jet_y_train, jet_test_pca, jet_y_test, verbose=False)
        print_training_history(cnn_history,
                               model_type=model_type,
                               N_components=N_components,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               loss_type='bce')
        if save_weights:
            CNN_train.save_weights()
        CNN_train.average_cnn_performance(jet_train_pca, jet_y_train, jet_test_pca, jet_y_test, n_trials=50)

if __name__ == "__main__":
    main()
