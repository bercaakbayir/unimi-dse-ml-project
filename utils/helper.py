import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def IQR(df, threshold=3, exclude_cols='quality'):
    df_copy = df.copy()

    # Exclude columns if specified
    if exclude_cols is not None:
        df_numeric = df_copy.select_dtypes(include='number').drop(columns=exclude_cols, errors='ignore')
    else:
        df_numeric = df_copy.select_dtypes(include='number')

    df = df_numeric

    Q1 = df[df.columns].quantile(0.25)
    Q3 = df[df.columns].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df[df.columns] < (Q1 - threshold * IQR)) | (
            df[df.columns] > (Q3 + threshold * IQR)
    )
    print("Outlier Percentages per Column")
    print((outliers.sum() / len(df)) * 100)


def cap_outliers_iqr(df, threshold=3, exclude_cols='quality'):
    df_capped = df.copy()

    # Select numeric columns and exclude specified ones if needed
    if exclude_cols is not None:
        numeric_cols = df_capped.select_dtypes(include='number').drop(columns=exclude_cols, errors='ignore').columns
    else:
        numeric_cols = df_capped.select_dtypes(include='number').columns

    Q1 = df_capped[numeric_cols].quantile(0.25)
    Q3 = df_capped[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_cap = Q1 - threshold * IQR
    upper_cap = Q3 + threshold * IQR

    for col in numeric_cols:
        df_capped[col] = df_capped[col].clip(lower=lower_cap[col], upper=upper_cap[col])

    return df_capped



def standard_scale(df, exclude_cols=['quality','wine_type_red', 'wine_type_white']):
    df_scaled = df.copy()

    if exclude_cols is None:
        exclude_cols = []  # make sure it's a list

    numeric_cols = df_scaled.select_dtypes(include='number').columns
    numeric_cols = numeric_cols.difference(exclude_cols)  # exclude_cols must be list-like here

    for col in numeric_cols:
        mean = df_scaled[col].mean()
        std = df_scaled[col].std()
        if std != 0:
            df_scaled[col] = (df_scaled[col] - mean) / std
        else:
            df_scaled[col] = 0

    return df_scaled


def skewness(arr):
    n = len(arr)
    mean = sum(arr) / n
    std = (sum((x - mean) ** 2 for x in arr) / (n - 1)) ** 0.5

    skewness_numer = sum(((x - mean) / std) ** 3 for x in arr)
    skewness = (n / ((n - 1) * (n - 2))) * skewness_numer

    return skewness



def boxcox_transform(x, lam):
    x = np.array(x)
    if np.any(x <= 0):
        raise ValueError("All values must be positive for Box-Cox transform.")
    if lam == 0:
        return np.log(x)
    else:
        return (x ** lam - 1) / lam


def boxcox_log_likelihood(x, lam):
    n = len(x)
    y = boxcox_transform(x, lam)
    var = np.var(y, ddof=1)
    log_lik = - (n / 2) * np.log(var) + (lam - 1) * np.sum(np.log(x))
    return log_lik


def find_best_lambda(x, lam_range=np.linspace(-2, 2, 100)):
    best_lambda = None
    best_log_lik = -np.inf

    for lam in lam_range:
        try:
            ll = boxcox_log_likelihood(x, lam)
            if ll > best_log_lik:
                best_log_lik = ll
                best_lambda = lam
        except Exception:
            continue
    return best_lambda


def boxcox_transformation(df, columns, skew_thresh=0.5):
    transformed_df = df.copy()
    lambdas = {}
    skipped_cols = []

    for col in columns:
        data = transformed_df[col].values
        col_skew = skewness(data)

        if col_skew > skew_thresh:
            min_val = np.min(data)
            if min_val <= 0:
                data = data + abs(min_val) + 1e-6

            lam = find_best_lambda(data)
            lambdas[col] = lam

            transformed_df[col] = boxcox_transform(data, lam)  # Replace original column
        else:
            skipped_cols.append(col)

    if skipped_cols:
        print(f"Skipped columns (skewness ≤ {skew_thresh}): {skipped_cols}")

    return transformed_df, lambdas


def VIF(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame X.

    Parameters:
        X (pd.DataFrame): A DataFrame containing only numeric independent variables.

    Returns:
        pd.DataFrame: A DataFrame with features and their corresponding VIF values.
    """
    X = X.copy()
    vif_data = []

    for i in range(X.shape[1]):
        y = X.iloc[:, i]
        X_others = X.drop(X.columns[i], axis=1)

        # Add intercept term
        X_others = X_others.copy()
        X_others['intercept'] = 1

        # Calculate coefficients using normal equation: β = (XᵀX)^(-1) Xᵀy
        X_matrix = X_others.values
        y_vector = y.values

        try:
            beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_vector
        except np.linalg.LinAlgError:
            vif = np.inf
            vif_data.append((X.columns[i], vif))
            continue

        y_hat = X_matrix @ beta
        ss_total = np.sum((y_vector - np.mean(y_vector)) ** 2)
        ss_res = np.sum((y_vector - y_hat) ** 2)
        r_squared = 1 - (ss_res / ss_total)

        # Compute VIF
        if r_squared >= 1.0:
            vif = np.inf
        else:
            vif = 1 / (1 - r_squared)

        vif_data.append((X.columns[i], vif))

    return pd.DataFrame(vif_data, columns=['Feature', 'VIF'])


def shannon_entropy(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum(
        [(count / n) * np.log((count / n)) for clas, count in classes]
    )  # shannon entropy
    return H / np.log(k)


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split data into training and test sets.

    Parameters:
    - X: features (DataFrame or numpy array)
    - y: target (Series or numpy array)
    - test_size: proportion of test set (e.g. 0.2 for 20%)
    - random_state: for reproducibility
    - shuffle: whether to shuffle the data before splitting

    Returns:
    - X_train, X_test, y_train, y_test
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.values

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.RandomState(seed=random_state)
        rng.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test




def plot_model_loss(model, vline_epoch=None, title=None, train_label="Train Loss", test_label="Test Loss", train_color='blue', test_color='orange'):
    """
    Plot training and test loss per epoch for a given model (Logistic Regression or SVM).

    Parameters:
    - model: Trained model instance, expected to have attributes: n_iters, train_losses, optionally test_losses.
    - vline_epoch: (int or None) Optional epoch number to highlight with a vertical dashed line.
    - title: (str or None) Title for the plot. If None, a default title will be generated based on model class name.
    - train_label: (str) Label for training loss line.
    - test_label: (str) Label for test loss line.
    - train_color: (str) Color for training loss line.
    - test_color: (str) Color for test loss line.
    """
    epochs = range(1, model.n_iters + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, model.train_losses, label=train_label, color=train_color)
    
    if hasattr(model, 'test_losses') and model.test_losses:
        plt.plot(epochs, model.test_losses, label=test_label, color=test_color)

    if vline_epoch is not None:
        plt.axvline(x=vline_epoch, color='red', linestyle='--', label=f'Epoch {vline_epoch}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if title is None:
        model_name = type(model).__name__
        title = f'{model_name} Loss per Epoch'

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
