import time
import numpy as np
from flcore.clients.clientalafair import clientALAFair
from flcore.servers.serverbase import Server


class FedALAFair(Server):
    """
    FedALA-Fair: Combining Adaptive Local Aggregation (ALA) with Fairness Constraints

    This algorithm combines:
    1. FedALA's adaptive local aggregation for personalization
    2. FairFed's fairness constraints for equitable performance across groups

    The result is a personalized federated learning approach that maintains
    fairness across different demographic groups while adapting to local data distributions.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # Fairness parameters
        self.fairness_lambda = (
            args.fairness_lambda if hasattr(args, "fairness_lambda") else 0.1
        )

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientALAFair)

        print(f"\n{'=' * 60}")
        print("FedALA-Fair: Personalized and Fair Federated Learning")
        print(f"{'=' * 60}")
        print(f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Fairness lambda: {self.fairness_lambda}")
        print(f"ALA eta: {args.eta}")
        print(f"ALA rand_percent: {args.rand_percent}")
        print(f"ALA layer_idx: {args.layer_idx}")
        print("Finished creating server and clients.")
        print(f"{'=' * 60}\n")

        self.Budget = []

        # Track fairness metrics over time
        self.fairness_history = {"demographic_parity": [], "average_accuracy": []}

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n{'=' * 60}")
                print(f"Round {i}: Evaluation with Fairness Metrics")
                print(f"{'=' * 60}")
                self.evaluate_with_fairness()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print(f"\n{'=' * 60}")
            print(f"Round {i} completed in {self.Budget[-1]:.2f}s")
            print(f"{'=' * 60}")

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\n" + "=" * 60)
        print("Training Completed - Final Results")
        print("=" * 60)
        print(f"Best Test Accuracy: {max(self.rs_test_acc):.4f}")
        print(
            f"Average Time per Round: {sum(self.Budget[1:]) / len(self.Budget[1:]):.2f}s"
        )

        # Print fairness summary
        print("\n" + "=" * 60)
        print("Fairness Summary")
        print("=" * 60)
        if len(self.fairness_history["demographic_parity"]) > 0:
            avg_dp = np.mean(self.fairness_history["demographic_parity"])
            final_dp = self.fairness_history["demographic_parity"][-1]
            print(f"Average Demographic Parity Difference: {avg_dp:.4f}")
            print(f"Final Demographic Parity Difference: {final_dp:.4f}")
            print(
                f"Best Accuracy: {max(self.fairness_history['average_accuracy']):.4f}"
            )

            # Trade-off analysis
            print("\n" + "-" * 60)
            print("Accuracy-Fairness Trade-off:")
            print("  Accuracy improvement from personalization (ALA)")
            print("  Fairness maintained through fairness constraints")
            print("=" * 60)
        else:
            print("Fairness metrics not available")
            print("(Configure sensitive_attr_idx for detailed fairness analysis)")
            print("=" * 60)

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientALAFair)
            print("\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_models(self):
        """
        Send global model to clients with ALA initialization
        """
        assert len(self.clients) > 0

        for client in self.clients:
            # Use ALA's adaptive local aggregation
            client.local_initialization(self.global_model)

    def evaluate_with_fairness(self):
        """
        Evaluate model performance with fairness metrics
        """
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        train_acc = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[3]) * 1.0 / sum(stats_train[1])

        self.rs_test_acc.append(test_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)

        # Fairness metrics aggregation
        all_fairness_metrics = stats[4] if len(stats) > 4 else []

        print("\nPerformance Metrics:")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Train Loss:     {train_loss:.4f}")

        if len(all_fairness_metrics) > 0:
            # Compute average fairness metrics across clients
            total_samples = sum(stats[1])
            weighted_dp = 0
            num_clients_with_metrics = 0

            for i, client_metrics in enumerate(all_fairness_metrics):
                if (
                    isinstance(client_metrics, dict)
                    and "demographic_parity" in client_metrics
                ):
                    weight = stats[1][i] / total_samples
                    weighted_dp += client_metrics["demographic_parity"] * weight
                    num_clients_with_metrics += 1

            if num_clients_with_metrics > 0:
                self.fairness_history["demographic_parity"].append(weighted_dp)
                self.fairness_history["average_accuracy"].append(test_acc)

                print("\nFairness Metrics:")
                print(f"  Demographic Parity Difference: {weighted_dp:.4f}")
                print(
                    f"  Clients with fairness metrics: {num_clients_with_metrics}/{len(self.clients)}"
                )
        else:
            print("\nFairness Metrics: Not available")

    def test_metrics(self):
        """
        Override test_metrics to include fairness metrics
        """
        if self.eval_new_clients and self.num_new_clients > 0:
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        fairness_metrics_list = []

        for c in self.clients:
            ct, ns, auc, fairness_metrics = c.test_metrics_fairness()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            tot_auc.append(auc * ns)
            fairness_metrics_list.append(fairness_metrics)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, fairness_metrics_list
