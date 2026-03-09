import numpy as np
import h5py
import os
from flcore.servers.serverala import FedALA


class FedALA_Fair(FedALA):
    """FedALA with fairness metrics tracking (EOD, AccGap, AccStd, AccWorst).

    Inherits the standard FedALA training loop (adaptive local aggregation)
    unchanged and only extends evaluate() and save_results() to record
    fairness metrics alongside accuracy and loss.
    """

    def __init__(self, args, times):
        super().__init__(args, times)
        self.rs_eod = []
        self.rs_acc_gap = []
        self.rs_acc_std = []
        self.rs_acc_worst = []

    def evaluate(self, acc=None, loss=None):
        super().evaluate(acc, loss)
        self._evaluate_fairness()

    def _evaluate_fairness(self):
        stats = []
        for c in self.clients:
            correct, total, _, fm = c.test_metrics_fairness()
            if total > 0:
                stats.append({"correct": correct, "total": total, **fm})

        if not stats:
            for lst in [
                self.rs_eod,
                self.rs_acc_gap,
                self.rs_acc_std,
                self.rs_acc_worst,
            ]:
                lst.append(float("nan"))
            return

        total_samples = sum(s["total"] for s in stats)

        # EOD: 全局聚合 TP/正类 计数后计算 TPR 差（论文公式 2）
        # 不能按客户端 total 加权 per-client EOD——无正类的大客户端（eod=0）
        # 权重大，会把整体 EOD 错误地拉向 0。
        total_tp_g0 = sum(s.get("n_tp_g0", 0) for s in stats)
        total_y1_g0 = sum(s.get("n_y1_g0", 0) for s in stats)
        total_tp_g1 = sum(s.get("n_tp_g1", 0) for s in stats)
        total_y1_g1 = sum(s.get("n_y1_g1", 0) for s in stats)
        tpr_g0 = total_tp_g0 / total_y1_g0 if total_y1_g0 > 0 else 0.0
        tpr_g1 = total_tp_g1 / total_y1_g1 if total_y1_g1 > 0 else 0.0
        weighted_eod = abs(tpr_g0 - tpr_g1)

        # AccGap: aggregate per-group counts then compute group-level accuracy
        total_g0_correct = sum(s.get("n_correct_g0", 0) for s in stats)
        total_g0 = sum(s.get("n_g0", 0) for s in stats)
        total_g1_correct = sum(s.get("n_correct_g1", 0) for s in stats)
        total_g1 = sum(s.get("n_g1", 0) for s in stats)
        acc_g0 = total_g0_correct / total_g0 if total_g0 > 0 else float("nan")
        acc_g1 = total_g1_correct / total_g1 if total_g1 > 0 else float("nan")
        acc_gap = (
            abs(acc_g0 - acc_g1)
            if not (np.isnan(acc_g0) or np.isnan(acc_g1))
            else float("nan")
        )

        # AccStd, AccWorst (10th percentile) across clients
        per_client_accs = [s["correct"] / s["total"] for s in stats]
        acc_std = float(np.std(per_client_accs))
        acc_worst = float(np.percentile(per_client_accs, 10))

        self.rs_eod.append(weighted_eod)
        self.rs_acc_gap.append(acc_gap)
        self.rs_acc_std.append(acc_std)
        self.rs_acc_worst.append(acc_worst)

        print(
            f"  [Fairness] EOD: {weighted_eod:.4f} | AccGap: {acc_gap:.4f} | "
            f"AccStd: {acc_std:.4f} | AccWorst(p10): {acc_worst:.4f}"
        )

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        os.makedirs(result_path, exist_ok=True)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
                hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss)
                hf.create_dataset("rs_eod", data=self.rs_eod)
                hf.create_dataset("rs_acc_gap", data=self.rs_acc_gap)
                hf.create_dataset("rs_acc_std", data=self.rs_acc_std)
                hf.create_dataset("rs_acc_worst", data=self.rs_acc_worst)
