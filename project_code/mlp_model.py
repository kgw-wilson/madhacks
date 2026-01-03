import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Define the MLP model
class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        learning_rate=0.001,  # Same lr from RouterBench paper
        evaluate_every=10,
        save_path=None,
    ):
        super(MLP, self).__init__()
        if (
            save_path is None
            or not save_path.endswith(".ckpt")
            or os.path.isdir(os.path.isdir(os.path.dirname(save_path)))
        ):
            raise RuntimeError(f"Bad path {save_path=}")
        self.save_path = save_path
        layers = []
        current_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.evaluate_every = evaluate_every

    def forward(self, x):
        return self.model(x)

    def train_model(self, X_train, y_train, X_eval, y_eval, epochs=10, batch_size=64):
        """
        Train the model using the provided training data and evaluate on eval data.

        Args:
            X_train (numpy.ndarray): Training feature data.
            y_train (numpy.ndarray): Training labels.
            X_eval (numpy.ndarray): Evaluation feature data.
            y_eval (numpy.ndarray): Evaluation labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        # Convert NumPy arrays to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_eval = torch.tensor(X_eval, dtype=torch.float32)
        y_eval = torch.tensor(y_eval, dtype=torch.long)

        # Create DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Early stop behavior
        max_accuracies = [0]
        evals_without_improvement = 0
        max_evals_without_improvement = 5

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Evaluation
            if epoch % self.evaluate_every == 0:
                accuracy = self.evaluate(X_eval, y_eval)
                if accuracy > max(max_accuracies):
                    max_accuracies.append(accuracy)
                    print(
                        f"Saving. Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Eval Accuracy: {accuracy:.4f}"
                    )
                    torch.save(self.state_dict(), self.save_path)
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement == max_evals_without_improvement:
                        print(f"Early stop at {max_evals_without_improvement=}.")
                        break

    def evaluate(self, X_eval, y_eval):
        """
        Evaluate the model on evaluation data.

        Args:
            X_eval (torch.Tensor): Evaluation feature data.
            y_eval (torch.Tensor): Evaluation labels.

        Returns:
            float: Accuracy of the model on the evaluation data.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = self(X_eval)
            _, predicted = torch.max(outputs, 1)
            total += y_eval.size(0)
            correct += (predicted == y_eval).sum().item()
        return correct / total
