# QCNN for JetNet Dataset
A qiskit-based utility to train a QCNN model on the Top Quark Tagging dataset. You can also
train an accordingly-shaped CNN model as a benchmark.

### Installation
    If you run on a local machine, we suggest to set up a venv for the dependencies:
    `python3 -m venv .venv`
    `source .venv/bin/activate`
    `pip3 install -r requirements.txt`
    If you run this program on a colab notebook, we suggest to install the dependencies
    contained in `requirements.txt` one by one.
### Execution
    You can find some demo config files in the `./configs/` folder to correctly configure
    the training hyperparameters. If you want to build some custom files of yours, please
    follow the format proposed.
    To launch an execution, run:
    `python3 main.py \`
    `--config_file [config_file] \`
    `--train_size [train_size] \`
    `--test_size [test_size] \`
    `--N_components [N_components] \`
    `--model_type [model_type] \`
    `--encoding_type [encoding_type] \`
    `--conv_type [conv_type] \`
    `--loss_type [loss_type] \`
    `--pretrained_weights_file [pretrained_weights_file] \`
    `--save_weights [save_weights]`
    for more info on the execution flags, run `python3 main.py -h`.
    
    Training history plots and saved trained weights are generated in the `./.plots/` and `./.pretrained/` folders.

### References
    [1] H. Elhag, T. Hartung, K. Jansen, L. Nagano, G. M. Pirina, and A. D. Tucci,
    “Quantum convolutional neural networks for jet images classification,” 2025.
    [2] I. Cong, S. Choi, and M. D. Lukin, “Quantum convolutional neural net-
    works,” Nature Physics, vol. 15, p. 1273–1278, Aug. 2019.
    5
    [3] J. T. M. R. G. Kasieczka, T. Plehn, “Top quark tagging reference dataset,”
    Mar. 2019.
    [4] R. Kansal, C. Pareja, Z. Hao, and J. Duarte, “JetNet: A Python package
    for accessing open datasets and benchmarking machine learning methods
    in high energy physics,” Journal of Open Source Software, vol. 8, no. 90,
    p. 5789, 2023.
    [5] T. S. Roy and A. H. Vijay, “A robust anomaly finder based on autoencoders,”
    2020.
    [6] S. Thanasilp, S. Wang, N. A. Nghiem, P. Coles, and M. Cerezo, “Subtleties
    in the trainability of quantum machine learning models,” Quantum Machine
    Intelligence, vol. 5, May 2023.
    [7] C. Lee, I. F. Araujo, D. Kim, J. Lee, S. Park, J.-Y. Ryu, and D. K. Park,
    “Optimizing quantum convolutional neural network architectures for arbi-
    trary data dimension,” Frontiers in Physics, vol. 13, Mar. 2025.
