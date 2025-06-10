import numpy as np
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, learning_rate=0.01, lambda_l2=0.00, lambda_l1=0.00, n_iters=1000, positive_label='good'):
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.n_iters = n_iters
        self.positive_label = positive_label
        self.w = None
        self.b = None
        
    def get_params(self, deep=True):
        """Return parameters as a dictionary (required for cloning)"""
        return {
            'learning_rate': self.learning_rate,
            'lambda_l2': self.lambda_l2,
            'lambda_l1': self.lambda_l1,
            'n_iters': self.n_iters,
            'positive_label':self.positive_label
        }
        
    def set_params(self, **params):
        """Set parameters (required for cloning)"""
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "Only binary classification is supported"

        # Convert labels to +1 / -1
        y_numeric = np.where(y == self.positive_label, 1, -1).astype(float)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_numeric[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization
                    self.w -= self.learning_rate * (2 * self.lambda_l2 * self.w + self.lambda_l1 * np.sign(self.w))
                else:
                    # Regularization + hinge loss gradient
                    self.w -= self.learning_rate * (
                                2 * self.lambda_l2 * self.w + self.lambda_l1 * np.sign(self.w) - y_numeric[idx] * x_i)
                    self.b -= self.learning_rate * y_numeric[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        raw_preds = np.sign(linear_output)
        return np.where(raw_preds == 1, self.positive_label,
                        [label for label in self.classes_ if label != self.positive_label][0])


class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, positive_label='good'):
        self.C = C
        self.gamma = gamma
        self.positive_label = positive_label
        self.kernel_name = kernel
        self.kernel = self._get_kernel(kernel)
        self.alphas = None
        self.b = None

    def _get_kernel(self, name):
        if name == 'linear':
            return lambda x, y: np.dot(x, y.T)
        elif name == 'poly':
            return lambda x, y: (1 + np.dot(x, y.T)) ** 2
        elif name == 'rbf':
            return lambda x, y: np.exp(-self.gamma * np.linalg.norm(x[:, None] - y, axis=2) ** 2)
        elif callable(name):
            return name
        else:
            raise ValueError("Unknown kernel function")

    def fit(self, X, y):
        m, n = X.shape
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "Only binary classification is supported"

        # Convert labels to +1 and -1
        y_numeric = np.where(y == self.positive_label, 1, -1).astype(float)

        # Compute the kernel matrix
        K = self.kernel(X, X)

        P = matrix(np.outer(y_numeric, y_numeric) * K)
        q = matrix(-np.ones(m))
        G = matrix(np.vstack([-np.eye(m), np.eye(m)]))
        h = matrix(np.hstack([np.zeros(m), np.ones(m) * self.C]))
        A = matrix(y_numeric.reshape(1, -1))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Support vectors have non-zero Lagrange multipliers
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y_numeric[sv]

        # Compute the bias term
        self.b = np.mean([
            y_k - np.sum(self.alphas * self.sv_y * self.kernel(np.array([x_k]), self.sv_X))
            for x_k, y_k in zip(self.sv_X, self.sv_y)
        ])

    def project(self, X):
        return np.sum(self.alphas * self.sv_y * self.kernel(X, self.sv_X), axis=1) + self.b

    def predict(self, X):
        raw_preds = np.sign(self.project(X))
        return np.where(raw_preds == 1, self.positive_label,
                        [label for label in self.classes_ if label != self.positive_label][0])


class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, l1=0.0, l2=0.0, positive_class="good"):
        """
        Parameters:
        - learning_rate: gradient step size
        - epochs: number of passes through training data
        - l1: L1 regularization strength
        - l2: L2 regularization strength
        - positive_class: label to treat as +1 (can be string like 'good')
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.positive_class = positive_class
        self.weights = None
        self.negative_class = None  # will be set during fit

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _encode_labels(self, y):
        """
        Convert original labels to {-1, +1} based on positive_class.
        Also sets self.negative_class if not already set.
        """
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("This implementation only supports binary classification.")

        if self.positive_class not in unique_classes:
            raise ValueError(f"Positive class '{self.positive_class}' not found in labels.")

        # Infer the other class
        self.negative_class = [cls for cls in unique_classes if cls != self.positive_class][0]

        return np.where(y == self.positive_class, 1, -1)

    def _loss_gradient(self, x, y):
        z = np.dot(self.weights, x)
        sig = self.sigmoid(-y * z)
        grad = -sig * y * x

        # Regularization
        grad += self.l2 * self.weights  # L2
        grad += self.l1 * np.sign(self.weights)  # L1

        return grad
    
    def get_params(self, deep=True):
        """Return parameters as a dictionary (required for cloning)"""
        return {
            'learning_rate': self.learning_rate,
            'epochs':self.epochs,
            'l2': self.l2,
            'l1': self.l1,
            'positive_class':self.positive_class
        }
        
    def set_params(self, **params):
        """Set parameters (required for cloning)"""
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y):
        m, n = X.shape
        y = self._encode_labels(y)
        self.weights = np.zeros(n)
        for epoch in range(self.epochs):
            for i in range(m):
                grad = self._loss_gradient(X[i], y[i])
                self.weights -= self.learning_rate * grad

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.where(probs >= 0.5, self.positive_class, self.negative_class)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


class KernelLogisticRegression:
    def __init__(self, kernel='rbf', gamma=1.0, degree=3, coef0=1.0,
                 learning_rate=0.1, epochs=1000, l2=0.0, positive_class=1):
        """
        Parameters:
        - kernel: 'linear', 'poly', or 'rbf'
        - gamma: kernel coefficient for rbf/poly
        - degree: degree for polynomial kernel
        - coef0: independent term for poly kernel
        - learning_rate: step size
        - epochs: number of training epochs
        - l2: L2 regularization strength
        - positive_class: label to treat as +1 (can be string)
        """
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.positive_class = positive_class

        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.negative_class = None

    def _encode_labels(self, y):
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("Only binary classification is supported.")
        if self.positive_class not in unique_classes:
            raise ValueError(f"Positive class '{self.positive_class}' not found in labels.")

        # Identify and store the negative class
        self.negative_class = [cls for cls in unique_classes if cls != self.positive_class][0]
        return np.where(y == self.positive_class, 1, -1)

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (self.gamma * X1 @ X2.T + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dist_sq = X1_sq - 2 * X1 @ X2.T + X2_sq
            return np.exp(-self.gamma * dist_sq)
        else:
            raise ValueError("Unsupported kernel type")

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        y_bin = self._encode_labels(y)
        m = X.shape[0]
        self.X_train = X
        self.y_train = y_bin
        self.alpha = np.zeros(m)

        K = self._kernel_function(X, X)

        for epoch in range(self.epochs):
            for i in range(m):
                z = np.dot(self.alpha * self.y_train, K[:, i])
                p = self.sigmoid(-self.y_train[i] * z)
                grad = -p * self.y_train[i] + self.l2 * self.alpha[i]
                self.alpha[i] -= self.learning_rate * grad

    def predict_proba(self, X):
        K = self._kernel_function(self.X_train, X)
        logits = np.dot(self.alpha * self.y_train, K)
        return self.sigmoid(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.where(probs >= 0.5, self.positive_class, self.negative_class)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


