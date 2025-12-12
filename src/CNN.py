import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class CNN_2by2(nn.Module):
    def __init__(self, out_channels, kernel_size, num_classes=2):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN_2by2, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=0)
        # Max pooling layer: 2x2 window, stride 1
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(out_channels, num_classes)
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        #x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        # x = self.softmax(x)
        return x


class CNN_2by3(nn.Module):
    def __init__(self, out_channels1, out_channels2, kernel_size, num_classes=2):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN_2by3, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels1,
                                      kernel_size=kernel_size, stride=1, padding=(1, 0))
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2,
                                      kernel_size=kernel_size, stride=1, padding=0)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(out_channels2, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        # x = self.pool(x)           # Apply max pooling
        x = self.fc1(x)            # Apply fully connected layer
        return x


class CNN_3by3(nn.Module):
    def __init__(self, out_channels1, out_channels2, kernel_size, num_classes=2):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN_3by3, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels1,
                                      kernel_size=kernel_size, stride=1, padding=0)
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2,
                                      kernel_size=kernel_size, stride=1, padding=0)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(out_channels2, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x


class CNN_training():

    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.tensor(data).float()
            self.labels = torch.tensor(labels).long() # Ensure labels are long for CrossEntropyLoss

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label


    def __init__(self, input_size, out_channels, kernel_size, device, pretrained_weights=None, **kwargs):
        self._reset(input_size, out_channels, kernel_size, device, pretrained_weights, **kwargs)

    def _reset(self, input_size, out_channels, kernel_size, device, pretrained_weights=None, **kwargs):
        self.input_size = input_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = device
        self.hyper = kwargs
        if input_size == 4:
            self.model = CNN_2by2(out_channels=out_channels, kernel_size=kernel_size,
                                          num_classes=2).to(device).float()
        elif input_size == 6:
            self.model = CNN_2by3(out_channels1=out_channels, out_channels2=out_channels,
                      kernel_size=kernel_size, num_classes=2).to(device).float()
        elif input_size == 9:
            self.model = CNN_3by3(out_channels1=out_channels, out_channels2=out_channels,
                      kernel_size=kernel_size, num_classes=2).to(device).float()
        else:
            raise NotImplementedError("NotImplementedError: no CNN model has been implemented to support this image size.")
        if pretrained_weights:
            self.model.load_state_dict(pretrained_weights)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyper["learning_rate"], weight_decay=self.hyper["L2_lambda"])

    def _get_dataloaders(self, X_data, y_data):
        if self.input_size == 4:
            image_shape = (-1, 1, 2, 2)
        elif self.input_size == 6:
            image_shape = (-1, 1, 2, 3)
        elif self.input_size == 9:
            image_shape = (-1, 1, 3, 3)
        dataset = self.CustomDataset(X_data.reshape(image_shape), y_data)
        dataloader = DataLoader(dataset, batch_size=self.hyper["batch_size"], shuffle=False)
        return dataloader

    def train(self, X_train_data, y_train_data, X_test_data=None, y_test_data=None, verbose=True):
        history = {
            "accuracy_tr" : [],
            "accuracy_ts" : [],
            "loss_tr" : [],
            "loss_ts" : []
          }
        train_dataloader = self._get_dataloaders(X_train_data, y_train_data)
        if (X_test_data is not None) and (y_test_data is not None):
            test_dataloader = self._get_dataloaders(X_test_data, y_test_data)
        l1_lambda = self.hyper["L1_lambda"] # Usa il lambda salvato

        for epoch in range(self.hyper["num_epochs"]):
            if verbose:
                print(f"Epoch [{epoch + 1}/{self.hyper['num_epochs']}]")
            for batch_index, (data, targets) in enumerate(tqdm(train_dataloader)):
                # Move data and targets to the device (GPU/CPU)
                data, targets = torch.tensor(data).float(), torch.tensor(targets).long()
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass: compute the model output
                scores = self.model(data)
                loss = self.criterion(scores, targets) # Standard CrossEntropyLoss

                # --- Regolarizzazione L1: Aggiunta manuale alla Loss ---
                l1_norm = torch.tensor(0.0).to(self.device)
                if l1_lambda > 0:
                    for name, param in self.model.named_parameters():
                        # Di solito si regolarizzano solo i pesi ('weight') non i bias ('bias')
                        if 'weight' in name:
                            l1_norm += torch.linalg.norm(param, ord=1) # Calcola la norma L1 del tensore
                    
                    loss = loss + l1_lambda * l1_norm
                # --------------------------------------------------------

                # Backward pass: compute the gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Optimization step: update the model parameters
                self.optimizer.step()
            x_tr, y_tr = torch.tensor(X_train_data).float(), torch.tensor(y_train_data).long()
            if (X_test_data is not None) and (y_test_data is not None):
                x_ts, y_ts = torch.tensor(X_test_data).float(), torch.tensor(y_test_data).long()
            accuracy_tr, loss_tr = self.check_accuracy(x_tr, y_tr, verbose=False)
            if (X_test_data is not None) and (y_test_data is not None):
                accuracy_ts, loss_ts = self.check_accuracy(x_ts, y_ts, verbose=False)
            history["accuracy_tr"].append(accuracy_tr)
            history["loss_tr"].append(loss_tr)
            if (X_test_data is not None) and (y_test_data is not None):
                history["accuracy_ts"].append(accuracy_ts)
                history["loss_ts"].append(loss_ts)
        return history

    def check_accuracy(self, X_data, y_data, verbose=True):
        """
        Checks the accuracy of the model on the given dataset loader.
        """
        # if loader.dataset.train:
        #     print("Checking accuracy on training data")
        # else:
        #     print("Checking accuracy on test data")

        loader = self._get_dataloaders(X_data, y_data)
        num_correct = 0
        num_samples = 0
        loss = 0
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass: compute the model output
                scores = self.model(x)
                _, predictions = scores.max(1)  # Get the index of the max log-probability
                num_correct += (predictions == y).sum()  # Count correct predictions
                num_samples += predictions.size(0)  # Count total samples
                loss += self.criterion(scores, y)

            # Calculate accuracy
            accuracy = float(num_correct) / float(num_samples) * 100
            if verbose:
                print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

        self.model.train()  # Set the model back to training mode
        return accuracy, loss


### -- AVERAGE CNN PERFORMANCE -- ###

    def average_cnn_performance(self, X_train_full, y_train_full, X_test_full, y_test_full, n_trials=50):
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        for i in range(n_trials):
            self._reset(self.input_size, self.out_channels, self.kernel_size, self.device, **self.hyper)

            self.train(X_train_full, y_train_full, verbose=False)
            tr_accuracy, tr_loss = self.check_accuracy(X_train_full, y_train_full, verbose=False)
            train_accuracy.append(tr_accuracy)
            train_loss.append(tr_loss)

            ts_accuracy, ts_loss = self.check_accuracy(X_test_full, y_test_full, verbose=False)
            test_accuracy.append(ts_accuracy)
            test_loss.append(ts_loss)

        avg_loss_train = np.mean(train_loss)
        avg_accuracy_train = np.mean(train_accuracy)

        std_loss_train = np.std(train_loss)
        std_accuracy_train = np.std(train_accuracy)

        avg_loss_test = np.mean(test_loss)
        avg_accuracy_test = np.mean(test_accuracy)

        std_loss_test = np.std(test_loss)
        std_accuracy_test = np.std(test_accuracy)

        print(f"Loss media sul set di train: {avg_loss_train:.4f} +- {std_loss_train:.4f}")
        print(f"Accuratezza media sul set di train: {avg_accuracy_train:.2f} +- {std_accuracy_train:.2f} %")
        print(f"\nLoss media sul set di test: {avg_loss_test:.4f} +- {std_loss_test:.4f}")
        print(f"Accuratezza media sul set di test: {avg_accuracy_test:.2f} +- {std_accuracy_test:.2f} %")

    def save_weights(self):
        weights_dir = ".pretrained"
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(weights_dir, f'pretrained_{self.input_size}_cnn_weights.pth'))
