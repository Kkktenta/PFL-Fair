import numpy as np
import os
import sys
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Adult/"


# Download and preprocess Adult dataset
def download_adult_dataset(data_path):
    """Download Adult dataset from UCI repository"""
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    test_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    )

    train_file = os.path.join(data_path, "adult.data")
    test_file = os.path.join(data_path, "adult.test")

    if not os.path.exists(train_file):
        print("Downloading Adult training data...")
        try:
            import urllib.request

            urllib.request.urlretrieve(train_url, train_file)
        except Exception as e:
            print(f"Failed to download training data: {e}")
            raise

    if not os.path.exists(test_file):
        print("Downloading Adult test data...")
        try:
            import urllib.request

            urllib.request.urlretrieve(test_url, test_file)
        except Exception as e:
            print(f"Failed to download test data: {e}")
            raise

    return train_file, test_file


def preprocess_adult_data(train_file, test_file):
    """Preprocess Adult dataset"""
    # Column names
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    # Load data
    train_data = pd.read_csv(
        train_file, names=column_names, sep=",\s*", engine="python", na_values="?"
    )
    test_data = pd.read_csv(
        test_file,
        names=column_names,
        sep=",\s*",
        engine="python",
        na_values="?",
        skiprows=1,
    )

    # Combine datasets
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Drop rows with missing values
    data = data.dropna()

    # Process target variable
    data["income"] = data["income"].apply(lambda x: 1 if ">50K" in x else 0)

    # Store sensitive attribute (for fairness evaluation)
    sensitive_attr = data["sex"].copy()

    # Select features
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    numerical_features = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    # Encode categorical features
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le

    # Separate features and labels
    X = data[numerical_features + categorical_features].values
    y = data["income"].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Get sensitive attribute index (sex) for fairness evaluation
    sex_idx = (numerical_features + categorical_features).index("sex")

    return X, y, sex_idx


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Download and preprocess Adult dataset
    rawdata_path = dir_path + "rawdata/"
    train_file, test_file = download_adult_dataset(rawdata_path)
    dataset_image, dataset_label, sex_idx = preprocess_adult_data(train_file, test_file)

    num_classes = len(set(dataset_label))
    print(f"Number of classes: {num_classes}")
    print(f"Number of samples: {len(dataset_label)}")
    print(f"Feature dimension: {dataset_image.shape[1]}")
    print(f"Sensitive attribute (sex) index: {sex_idx}")

    # For Adult dataset, we use non-IID partition based on sensitive attributes
    # to simulate realistic federated learning scenarios with fairness concerns
    if niid:
        # Partition data based on sensitive attribute to create non-IID distribution
        # This simulates real-world scenarios where different clients have
        # different demographic distributions
        partition = "dir" if partition == "-" else partition

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=2,
    )
    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
    )

    print("\nAdult dataset generated successfully!")
    print(f"Total clients: {num_clients}")
    print(f"Non-IID: {niid}")
    print(f"Balance: {balance}")
    print(f"Partition: {partition}")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
