import numpy as np
from cvxopt import matrix, solvers

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iters=100, l1=0.0, l2=0.0, positive_class="good",
                 kernel='linear', gamma=1.0, degree=3, coef0=1.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l1 = l1
        self.l2 = l2
        self.positive_class = positive_class
        self.negative_class = None
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.weights = None
        self.X_train = None

        # Metrics
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _encode_labels(self, y):
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("This implementation only supports binary classification.")
        if self.positive_class not in unique_classes:
            raise ValueError(f"Positive class '{self.positive_class}' not found in labels.")
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
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'n_iters': self.n_iters,
            'l1': self.l1,
            'l2': self.l2,
            'positive_class': self.positive_class,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def fit(self, X, y, X_test=None, y_test=None, silence=False):
        y = self._encode_labels(y)
        if X_test is not None and y_test is not None:
            y_test = self._encode_labels(y_test)

        self.X_train = X
        K = self._kernel_function(X, X)
        self.weights = np.zeros(K.shape[0])

        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        for epoch in range(self.n_iters):
            z = np.dot(K, self.weights)
            y_pred = self.sigmoid(z)

            m = K.shape[0]
            y_binary = (y + 1) / 2  # convert labels from {-1,1} to {0,1}
            errors = y_pred - y_binary  # predicted - true

            grad = np.dot(K.T, errors) / m

            grad += self.l2 * self.weights
            grad += self.l1 * np.sign(self.weights)

            self.weights -= self.learning_rate * grad

            z = y * np.dot(K, self.weights)
            train_loss = np.mean(np.log(1 + np.exp(-z)))
            train_acc = self.accuracy(X, np.where(y == 1, self.positive_class, self.negative_class))
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            if X_test is not None and y_test is not None:
                K_test = self._kernel_function(X_test, self.X_train)
                z_test = np.dot(K_test, self.weights)
                test_loss = -np.mean(np.log(self.sigmoid(y_test * z_test)))
                test_acc = self.accuracy(X_test, np.where(y_test == 1, self.positive_class, self.negative_class))
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_acc)

                if not silence:
                    print(f"[Epoch {epoch + 1}/{self.n_iters}] "
                          f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} || "
                          f"Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
            else:
                if not silence:
                    print(f"[Epoch {epoch + 1}/{self.n_iters}] "
                          f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

    def predict_proba(self, X):
        K = self._kernel_function(X, self.X_train)
        return self.sigmoid(np.dot(K, self.weights))

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.where(probs >= 0.5, self.positive_class, self.negative_class)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    def get_feature_importance(self):
        if self.kernel != 'linear':
            raise ValueError("Feature importance with linear weights only applies for linear kernel.")

        feature_importance = self.X_train.T @ self.weights  # shape (n_features,)

        return feature_importance



    


class SVM:
    def __init__(self, learning_rate=0.001, lambda_l2=0.00, lambda_l1=0.00, n_iters=100, positive_label='good', kernel='linear', C=1.0, gamma=1.0):
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.n_iters = n_iters
        self.positive_label = positive_label
        
        # Kernel related params
        self.kernel_name = kernel
        self.C = C  
        self.gamma = gamma
        
        self.w = None
        self.b = None
        self.alphas = None
        self.sv_X = None
        self.sv_y = None
        
        # Select kernel function
        self.kernel = self._get_kernel(kernel)
    
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
            raise ValueError(f"Unknown kernel function: {name}")
            
    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'positive_label': self.positive_label,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'n_iters': self.n_iters,
            'C': self.C
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        if 'kernel' in params:
            self.kernel = self._get_kernel(self.kernel_name)
        return self


    def fit(self, X_train, y_train, X_test=None, y_test=None, silence=False):
        self.classes_ = np.unique(y_train)
        assert len(self.classes_) == 2, "Only binary classification is supported"
        y_numeric = np.where(y_train == self.positive_label, 1, -1).astype(float)

        if self.kernel_name == 'linear':
            n_samples, n_features = X_train.shape
            self.w = np.zeros(n_features)
            self.b = 0

            self.train_losses = []
            self.test_losses = []
            self.train_accuracies = []
            self.test_accuracies = []

            for epoch in range(1, self.n_iters + 1):
                eta = self.learning_rate / (1 + 0.01 * epoch) 

                for idx, x_i in enumerate(X_train):
                    condition = y_numeric[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                    if condition:
                        self.w -= eta * (2 * self.lambda_l2 * self.w + self.lambda_l1 * np.sign(self.w))
                    else:
                        self.w -= eta * (2 * self.lambda_l2 * self.w + self.lambda_l1 * np.sign(self.w) - y_numeric[idx] * x_i)
                        self.b -= eta * y_numeric[idx]

                train_loss = self._hinge_loss(X_train, y_numeric)
                train_acc = self._accuracy(X_train, y_train)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)

                if X_test is not None and y_test is not None:
                    y_test_numeric = np.where(y_test == self.positive_label, 1, -1).astype(float)
                    test_loss = self._hinge_loss(X_test, y_test_numeric)
                    test_acc = self._accuracy(X_test, y_test)
                    self.test_losses.append(test_loss)
                    self.test_accuracies.append(test_acc)
                    if not silence:
                        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Accuracy={train_acc:.4f} | Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")

        else:
            m, n = X_train.shape
            K = self.kernel(X_train, X_train)
            P = matrix(np.outer(y_numeric, y_numeric) * K)
            q = matrix(-np.ones(m))
            G = matrix(np.vstack([-np.eye(m), np.eye(m)]))
            h = matrix(np.hstack([np.zeros(m), np.ones(m) * self.C]))
            A = matrix(y_numeric.reshape(1, -1))
            b = matrix(np.zeros(1))

            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.ravel(solution['x'])

            sv = alphas > 1e-5
            self.alphas = alphas[sv]
            self.sv_X = X_train[sv]
            self.sv_y = y_numeric[sv]

            self.b = np.mean([
                y_k - np.sum(self.alphas * self.sv_y * self.kernel(np.array([x_k]), self.sv_X))
                for x_k, y_k in zip(self.sv_X, self.sv_y)
            ])

            if not silence:
                print(f"Training done. Number of support vectors: {len(self.alphas)}")

    def predict(self, X):
        if self.kernel_name == 'linear':
            linear_output = np.dot(X, self.w) + self.b
            raw_preds = np.sign(linear_output)
        else:
            kernel_result = np.sum(self.alphas * self.sv_y * self.kernel(X, self.sv_X), axis=1) + self.b
            raw_preds = np.sign(kernel_result)

        return np.where(raw_preds == 1, self.positive_label,
                        [label for label in self.classes_ if label != self.positive_label][0])
    
    def predict_proba(self, X):
        """
        Returns probability estimates with explicit control over positive class.
        Ensures probabilities align with the intended positive label ('good').
        """
        if self.kernel_name == 'linear':
            decision = np.dot(X, self.w) + self.b
        else:
            decision = np.sum(self.alphas * self.sv_y * self.kernel(X, self.sv_X), axis=1) + self.b

        probs = 1 / (1 + np.exp(-decision))

        proba = np.zeros((len(probs), 2))

        if self.positive_label == 'good':
            proba[:, 1] = probs   # P(good)
            proba[:, 0] = 1 - probs  # P(not good)
        else:
            proba[:, 0] = probs   # P(original positive class)
            proba[:, 1] = 1 - probs  # P(good)
            self.classes_ = np.array([self.positive_label, 'good'])  # Update class labels

        return proba

    def _hinge_loss(self, X, y_numeric):
        distances = 1 - y_numeric * (np.dot(X, self.w) + self.b)
        distances = np.maximum(0, distances)  # hinge loss
        hinge_loss = np.mean(distances)

        l2_term = self.lambda_l2 * np.sum(self.w ** 2)
        l1_term = self.lambda_l1 * np.sum(np.abs(self.w))

        return hinge_loss + l2_term + l1_term

    def _accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
    
    def get_feature_importance(self, feature_names=None):
        if self.kernel_name != 'linear':
            raise ValueError("Feature importance is only available for linear kernel.")

        importance = np.abs(self.w)
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        return importance
    
    

