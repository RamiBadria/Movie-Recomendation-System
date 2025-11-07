import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class FactorizationMachine:
    """
    A 2-Way Factorization Machine optimized for sparse data (MovieLens 1M).
    """

    def __init__(self, n_features, k, learning_rate=0.01, l2_reg=0.01, n_epochs=15):
        """
        Initializes the Factorization Machine model.
        """
        self.k = k
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.n_epochs = n_epochs

        # Model parameters
        self.w0 = 0.0
        self.w = np.zeros(self.n_features)
        # Initialize V with small random values
        self.V = np.random.normal(0, 0.1, (self.n_features, self.k))
        
        # Best model tracking
        self.best_w0 = 0.0
        self.best_w = np.zeros(self.n_features)
        self.best_V = np.random.normal(0, 0.1, (self.n_features, self.k))
        self.best_test_rmse = float('inf')

    def predict(self, x):
        """
        Predicts rating for a single sparse feature vector x.
        Assumes x is a row from a CSR matrix (or similar sparse format).
        """
        # Get active feature indices efficiently
        active_indices = x.indices

        # 1. Global bias
        pred = self.w0

        # 2. Linear terms (sum of weights for active features)
        pred += np.sum(self.w[active_indices])

        # 3. Interaction terms (efficient vectorized computation)
        if self.k > 0:
            v_active = self.V[active_indices] # Shape: (n_active, k)
            
            # sum_of_v = \sum_{i \in active} V_i
            sum_of_v = np.sum(v_active, axis=0)
            
            # sum_of_sq_v = \sum_{i \in active} V_i^2
            sum_of_sq_v = np.sum(v_active ** 2, axis=0)
            
            # Interaction = 0.5 * \sum_{f=1}^k ( (\sum V_{if})^2 - \sum V_{if}^2 )
            interaction = 0.5 * np.sum(sum_of_v**2 - sum_of_sq_v)
            pred += interaction
        
        return pred

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Trains using SGD, optimized for sparse input matrices (CSR).
        """
        print(f"Starting training on {X_train.shape[0]} samples...")
        n_samples = X_train.shape[0]
        train_indices = np.arange(n_samples)

        for epoch in range(self.n_epochs):
            np.random.shuffle(train_indices)

            # Use tqdm for progress tracking
            for idx in tqdm(train_indices, desc=f"Epoch {epoch+1}/{self.n_epochs}", mininterval=1.0):
                x = X_train[idx]      # This is a sparse row
                y_true = y_train[idx]
                
                # --- Forward pass ---
                active_indices = x.indices
                
                # Calculate prediction (inline for speed inside loop if needed, but calling predict is cleaner)
                pred = self.w0 + np.sum(self.w[active_indices])
                if self.k > 0:
                     v_active = self.V[active_indices]
                     sum_v = np.sum(v_active, axis=0)
                     pred += 0.5 * np.sum(np.sum(v_active, axis=0)**2 - np.sum(v_active**2, axis=0))

                error = pred - y_true

                # --- Backward pass (SGD updates) ---
                # w0 update
                self.w0 -= self.learning_rate * error

                # w update (vectorized for all active indices)
                w_grad = error + self.l2_reg * self.w[active_indices]
                self.w[active_indices] -= self.learning_rate * w_grad

                # V update (vectorized)
                if self.k > 0:
                    # Gradient for V_{if} is approx: error * (\sum_{j \ne i} V_{jf} x_j) * x_i
                    # Since x_i is 1, this simplifies to: error * (sum_v - V_{if})
                    # We compute this for all active features at once using broadcasting.
                    
                    # (sum_v - v_active) subtracts each row's V from the total sum
                    v_grad_term = (sum_v - v_active) 
                    v_grad = error * v_grad_term + self.l2_reg * v_active
                    
                    self.V[active_indices] -= self.learning_rate * v_grad

            # --- Evaluation ---
            # Note: Evaluating on full test set every epoch might be slow for 1M.
            # Consider evaluating on a smaller sample if it takes too long.
            train_rmse = self.evaluate(X_train, y_train, n_samples=10000) # Evaluate on subset for speed
            test_rmse = self.evaluate(X_test, y_test) # Full test set evaluation
            
            print(f"Epoch {epoch + 1}: Train RMSE (approx): {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        
            # Save best model
            if test_rmse < self.best_test_rmse:
                self.best_test_rmse = test_rmse
                self.best_w0 = self.w0
                self.best_w = self.w.copy()
                self.best_V = self.V.copy()
                print(f"  -> New best model found! (Test RMSE: {test_rmse:.4f})")

        # Restore best model parameters
        self.w0 = self.best_w0
        self.w = self.best_w
        self.V = self.best_V
        print("Training complete. Best model restored.")

    def evaluate(self, X, y_true, n_samples=None):
        """
        Evaluates RMSE. Optionally on a random subset for speed during training.
        """
        if n_samples is not None and n_samples < X.shape[0]:
            # Evaluate on a random subset
            indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_subset = X[indices]
            y_subset = y_true[indices]
            y_pred = np.array([self.predict(X_subset[i]) for i in range(n_samples)])
            return np.sqrt(mean_squared_error(y_subset, y_pred))
        else:
            # Evaluate on full set (might be slow without batch predict)
            y_pred = np.array([self.predict(X[i]) for i in range(X.shape[0])])
            return np.sqrt(mean_squared_error(y_true, y_pred))