import numpy as np
import time
import torch
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA


class clientALAFair(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # ALA parameters
        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        train_data = read_client_data(
            self.dataset, self.id, is_train=True, few_shot=self.few_shot
        )
        self.ALA = ALA(
            self.id,
            self.loss,
            train_data,
            self.batch_size,
            self.rand_percent,
            self.layer_idx,
            self.eta,
            self.device,
        )

        # Fairness parameters
        self.fairness_lambda = (
            args.fairness_lambda if hasattr(args, "fairness_lambda") else 0.1
        )
        self.sensitive_attr_idx = (
            args.sensitive_attr_idx if hasattr(args, "sensitive_attr_idx") else -1
        )

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)

                # Standard classification loss
                classification_loss = self.loss(output, y)

                # Fairness loss (from FairFed)
                fairness_loss = self.compute_fairness_loss(x, output, y)

                # Total loss combining classification and fairness
                total_loss = classification_loss + self.fairness_lambda * fairness_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def local_initialization(self, received_global_model):
        """
        ALA adaptive local aggregation (from FedALA)
        """
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def compute_fairness_loss(self, x, output, y):
        """
        Compute fairness loss based on Demographic Parity
        (Inherited from FairFed implementation)
        """
        try:
            predictions = torch.softmax(output, dim=1)[:, 1]

            # Variance-based fairness for balanced predictions
            fairness_loss = torch.var(predictions)

            # If sensitive attribute is available, use group-specific fairness
            # if self.sensitive_attr_idx >= 0 and x.ndim == 2:
            #     sensitive_attr = x[:, self.sensitive_attr_idx]
            #     group_0_mask = (sensitive_attr < 0.5)
            #     group_1_mask = (sensitive_attr >= 0.5)
            #
            #     if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
            #         pred_group_0 = predictions[group_0_mask].mean()
            #         pred_group_1 = predictions[group_1_mask].mean()
            #         fairness_loss = torch.abs(pred_group_0 - pred_group_1)

            return fairness_loss

        except Exception:
            return torch.tensor(0.0, device=output.device)

    def test_metrics_fairness(self):
        """
        Compute test metrics including fairness metrics
        """
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0

        # Fairness metrics
        all_predictions = []
        all_labels = []
        all_sensitive = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # Store for fairness computation
                predictions = torch.argmax(output, dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(y.cpu().numpy())

                # Extract sensitive attribute if available
                if self.sensitive_attr_idx >= 0 and x.ndim == 2:
                    sensitive = x[:, self.sensitive_attr_idx]
                    all_sensitive.append(sensitive.cpu().numpy())

        accuracy = test_acc / test_num

        # Compute fairness metrics if sensitive attributes available
        fairness_metrics = {}
        if len(all_sensitive) > 0:
            all_predictions = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)
            all_sensitive = np.concatenate(all_sensitive)

            # Demographic Parity Difference
            group_0_mask = all_sensitive < 0.5
            group_1_mask = all_sensitive >= 0.5

            if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
                positive_rate_group_0 = (all_predictions[group_0_mask] == 1).mean()
                positive_rate_group_1 = (all_predictions[group_1_mask] == 1).mean()
                demographic_parity = abs(positive_rate_group_0 - positive_rate_group_1)

                fairness_metrics["demographic_parity"] = demographic_parity
                fairness_metrics["positive_rate_group_0"] = positive_rate_group_0
                fairness_metrics["positive_rate_group_1"] = positive_rate_group_1

        return test_acc, test_num, 0, fairness_metrics
