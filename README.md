# QCNN for JetNet Dataset
A QCNN implementation to be tested on the Top Quark Tagging dataset.

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
