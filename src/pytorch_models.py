"""
PyTorch Neural Network Models for Credit Scoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


class TabularDataset(Dataset):
    """Custom Dataset for tabular data"""
    def __init__(self, X, y):
        # Explicitly use float32 for MPS compatibility
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeedForwardNN(nn.Module):
    """
    Feed-forward neural network for binary classification
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(FeedForwardNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepNN(nn.Module):
    """
    Deeper neural network with residual connections
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 256, 128, 64], dropout=0.3):
        super(DeepNN, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Middle layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        return self.output_layer(x)


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible PyTorch classifier wrapper
    """
    def __init__(self,
                 model_type='feedforward',
                 hidden_dims=[256, 128, 64],
                 dropout=0.3,
                 learning_rate=0.001,
                 batch_size=256,
                 epochs=50,
                 early_stopping_patience=5,
                 device=None,
                 verbose=True,
                 pos_weight=1.0):

        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.verbose = verbose
        self.pos_weight = pos_weight
        self.model = None
        self.input_dim = None
        self.history = {'train_loss': [], 'val_loss': []}

        # Auto-detect device
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon GPU
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')  # NVIDIA GPU
            else:
                self.device = torch.device('cpu')

    def _build_model(self, input_dim):
        """Build the neural network model"""
        if self.model_type == 'feedforward':
            model = FeedForwardNN(input_dim, self.hidden_dims, self.dropout)
        elif self.model_type == 'deep':
            model = DeepNN(input_dim, self.hidden_dims, self.dropout)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return model.to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train the model

        Args:
            X: Training features (numpy array or pandas DataFrame)
            y: Training labels (numpy array or pandas Series)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Ensure float32 for MPS compatibility
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.input_dim = X.shape[1]

        # Build model
        self.model = self._build_model(self.input_dim)

        # Create datasets
        train_dataset = TabularDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Setup loss and optimizer
        pos_weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}') if self.verbose else train_loader

            for batch_X, batch_y in progress_bar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                optimizer.zero_grad()

                # Forward pass (remove sigmoid since BCEWithLogitsLoss includes it)
                outputs = self.model.network[:-1](batch_X)  # Get output before sigmoid
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if self.verbose and isinstance(progress_bar, tqdm):
                    progress_bar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)

            # Validation phase
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, criterion)
                self.history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

                if self.verbose:
                    print(f'Epoch {epoch+1}/{self.epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}')

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if self.verbose:
                            print(f'Early stopping at epoch {epoch+1}')
                        # Restore best model
                        self.model.load_state_dict(self.best_model_state)
                        break
            else:
                if self.verbose:
                    print(f'Epoch {epoch+1}/{self.epochs} - Loss: {avg_train_loss:.4f}')

        return self

    def _validate(self, X_val, y_val, criterion):
        """Validate the model"""
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        if hasattr(y_val, 'values'):
            y_val = y_val.values

        # Ensure float32 for MPS compatibility
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        val_dataset = TabularDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                outputs = self.model.network[:-1](batch_X)  # Get output before sigmoid
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Features (numpy array or pandas DataFrame)

        Returns:
            Probabilities for each class [P(class=0), P(class=1)]
        """
        if hasattr(X, 'values'):
            X = X.values

        # Ensure float32 for MPS compatibility
        X = np.array(X, dtype=np.float32)

        self.model.eval()

        dataset = TabularDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                predictions.append(outputs.cpu().numpy())

        predictions = np.vstack(predictions).flatten()

        # Return probabilities for both classes
        return np.column_stack([1 - predictions, predictions])

    def predict(self, X, threshold=0.5):
        """
        Predict class labels

        Args:
            X: Features
            threshold: Decision threshold

        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        """
        Score method for sklearn compatibility (used in cross_val_score)
        Returns ROC AUC score
        """
        from sklearn.metrics import roc_auc_score
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device,
            'verbose': self.verbose,
            'pos_weight': self.pos_weight
        }

    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_pytorch_model(model_type='feedforward', **kwargs):
    """
    Factory function to create PyTorch models

    Args:
        model_type: 'feedforward' or 'deep'
        **kwargs: Additional arguments for PyTorchClassifier

    Returns:
        PyTorchClassifier instance
    """
    return PyTorchClassifier(model_type=model_type, **kwargs)
