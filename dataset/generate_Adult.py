import numpy as np
import os
import sys
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.dataset_utils import check, save_file, alpha

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
    print(f"Partition alpha (demographic heterogeneity): {alpha}")

    # ── Partition by sensitive attribute (A = sex) ──────────────────────────
    # FairFed paper: heterogeneous A across clients, each client has both Y=0
    # and Y=1 so FedAvg can converge but learns a globally-biased model.
    #
    # Implementation: call PFLlib's separate_data() with A (sex) as the
    # "label" for Dirichlet partitioning.  Y labels travel with X because we
    # augment X with Y as an extra column before partitioning then strip it.
    # This reuses PFLlib's balancing constraint → no tiny / empty clients.
    #
    # clientbase.py binarises sex as (feature >= 0) after StandardScaler;
    # we use the same threshold so partition and evaluation are consistent.
    # ── Partition by sensitive attribute (A = sex) ──────────────────────────
    # The FairFed paper makes clients heterogeneous in A (demographic
    # composition) while each client has both Y=0 and Y=1 so FedAvg can
    # converge but learns a globally-biased model → EOD ≈ −0.17.
    #
    # We implement a direct single-pass Dirichlet partition (without the
    # PFLlib separate_data() retry loop, which never terminates for
    # alpha=0.1 with K=2 binary groups and N=45 k samples).
    #
    # clientbase.py binarises sex as (feature >= 0) after StandardScaler;
    # we use the same threshold so partition and evaluation are consistent.
    X_parts, y_parts, A_parts, statistic = _partition_by_sensitive(
        dataset_image, dataset_label, sex_idx, num_clients, alpha, niid
    )

    from sklearn.model_selection import train_test_split as _tts

    train_ratio = 0.75
    train_data, test_data = [], []
    num_samples = {"train": [], "test": []}
    for k in range(num_clients):
        X_tr, X_te, y_tr, y_te = _tts(
            X_parts[k], y_parts[k], train_size=train_ratio, shuffle=True
        )
        train_data.append({"x": X_tr, "y": y_tr})
        test_data.append({"x": X_te, "y": y_te})
        num_samples["train"].append(len(y_tr))
        num_samples["test"].append(len(y_te))

    print("Total number of samples:", sum(num_samples["train"] + num_samples["test"]))
    print("The number of train samples:", num_samples["train"])
    print("The number of test samples:", num_samples["test"])

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
    print(f"Non-IID: {niid}  (heterogeneous by sensitive attribute A=sex)")
    print(f"Alpha: {alpha}")


def _partition_by_sensitive(X, y, sensitive_idx, num_clients, alpha, niid):
    """Partition by sensitive attribute (sex) using Dirichlet(alpha).

    Each client gets a different female/male ratio (heterogeneous A), but
    ALL clients receive both Y=0 and Y=1 because Y and sex are only
    correlated, not deterministic.  This matches the FairFed paper setup.

    Uses a single Dirichlet draw per group (no retry loop) with a
    post-processing step that steals samples from large clients to ensure
    every client has at least MIN_TOTAL samples.
    """
    MIN_TOTAL = 50  # minimum total samples per client
    MIN_PER_GROUP = 30  # minimum samples per sensitive group per client (full dataset)
    # After 75% train/test split, each group has at least ~22 training samples.
    # MIN_PER_GROUP ensures every client has both A=0 and A=1 samples so that:
    # (1) clientfairfed can compute F_k (not fall back to acc gap)
    # (2) FairFed/RW reweighting is meaningful (both groups present in batches)
    # The FairFed paper (K=5) shows minimum ~32 minority samples per client.

    # Binarise sex consistently with clientbase.py
    A = (X[:, sensitive_idx] >= 0).astype(int)  # 0=female, 1=male
    idx_a0 = np.where(A == 0)[0]
    np.random.shuffle(idx_a0)  # female
    idx_a1 = np.where(A == 1)[0]
    np.random.shuffle(idx_a1)  # male
    n_a0, n_a1 = len(idx_a0), len(idx_a1)

    if not niid:
        # IID: equal split of each group
        counts_a0 = np.full(num_clients, n_a0 // num_clients, dtype=int)
        counts_a0[-1] = n_a0 - counts_a0[:-1].sum()
        counts_a1 = np.full(num_clients, n_a1 // num_clients, dtype=int)
        counts_a1[-1] = n_a1 - counts_a1[:-1].sum()
    else:
        # Non-IID: Dirichlet(alpha) over each group independently.
        # Low alpha → most clients get data dominated by one sex group;
        # high alpha → roughly uniform composition.
        props_a0 = np.random.dirichlet(np.repeat(alpha, num_clients))
        props_a1 = np.random.dirichlet(np.repeat(alpha, num_clients))
        counts_a0 = np.maximum(np.floor(props_a0 * n_a0).astype(int), 0)
        counts_a1 = np.maximum(np.floor(props_a1 * n_a1).astype(int), 0)
        # fix rounding: make sums exact
        counts_a0[-1] = max(0, n_a0 - counts_a0[:-1].sum())
        counts_a1[-1] = max(0, n_a1 - counts_a1[:-1].sum())

        def _steal_from_rich(counts, needy_k, need, min_keep):
            """Steal `need` samples for group from the largest donor clients."""
            remaining = need
            for donor in np.argsort(counts)[::-1]:
                if donor == needy_k or counts[donor] <= min_keep:
                    continue
                give = min(remaining, counts[donor] - min_keep)
                counts[needy_k] += give
                counts[donor] -= give
                remaining -= give
                if remaining <= 0:
                    break
            return counts

        # Step 1: ensure MIN_PER_GROUP for each group independently
        for k in range(num_clients):
            if counts_a0[k] < MIN_PER_GROUP:
                need = MIN_PER_GROUP - counts_a0[k]
                counts_a0 = _steal_from_rich(counts_a0, k, need, MIN_PER_GROUP)
            if counts_a1[k] < MIN_PER_GROUP:
                need = MIN_PER_GROUP - counts_a1[k]
                counts_a1 = _steal_from_rich(counts_a1, k, need, MIN_PER_GROUP)

        # Step 2: ensure MIN_TOTAL per client (total samples, both groups combined)
        totals = counts_a0 + counts_a1
        for k in range(num_clients):
            deficit = MIN_TOTAL - totals[k]
            if deficit <= 0:
                continue
            for donor in np.argsort(totals)[::-1]:
                if donor == k or totals[donor] <= MIN_TOTAL:
                    continue
                can_give = min(deficit, totals[donor] - MIN_TOTAL)
                ratio = counts_a0[donor] / max(totals[donor], 1)
                give_a0 = min(round(can_give * ratio), counts_a0[donor])
                give_a1 = min(can_give - give_a0, counts_a1[donor])
                give_a0 = can_give - give_a1
                give_a0 = min(give_a0, counts_a0[donor])
                counts_a0[k] += give_a0
                counts_a0[donor] -= give_a0
                counts_a1[k] += give_a1
                counts_a1[donor] -= give_a1
                totals[k] += give_a0 + give_a1
                totals[donor] -= give_a0 + give_a1
                deficit -= give_a0 + give_a1
                if deficit <= 0:
                    break

    X_clients, y_clients, A_clients, statistic = [], [], [], []
    ptr0, ptr1 = 0, 0
    for k in range(num_clients):
        n0, n1 = int(counts_a0[k]), int(counts_a1[k])
        idx = np.concatenate(
            [
                idx_a0[ptr0 : ptr0 + n0],
                idx_a1[ptr1 : ptr1 + n1],
            ]
        )
        ptr0 += n0
        ptr1 += n1
        Xk, yk = X[idx], y[idx]
        Ak = A[idx]
        stat_k = [(int(v), int((yk == v).sum())) for v in sorted(np.unique(yk))]
        X_clients.append(Xk)
        y_clients.append(yk)
        A_clients.append(Ak)
        statistic.append(stat_k)
        total = len(yk)
        print(
            f"Client {k}\t Size: {total}"
            f"\t female={n0}({100 * n0 // max(total, 1)}%)"
            f"\t male={n1}({100 * n1 // max(total, 1)}%)"
            f"\t Y={np.unique(yk)}"
        )
        print(f"\t\t Samples of labels: {stat_k}")
        print("-" * 50)

    return X_clients, y_clients, A_clients, statistic


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
