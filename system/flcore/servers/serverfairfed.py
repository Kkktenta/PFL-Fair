import time
import random
import numpy as np
import h5py
import os
from flcore.clients.clientfairfed import clientFairFed
from flcore.servers.serverbase import Server


class FairFed(Server):
    """
    FairFed: Enabling Group Fairness in Federated Learning (Ezzeldin et al., NeurIPS 2023).

    Implements Algorithm 1 (tracking EOD) exactly as described in the paper.

    Each communication round has three phases:

      Step 1 — ClientLocalMetrics:
        Clients evaluate the current global model on local training data and
        return per-group TPR counts.  The server aggregates these via Formula (7)
        to obtain F_global^t and Acc^t.

      Step 2 — ClientMetricGap:
        Server broadcasts F_global and Acc; each client computes Δ_k via
        the piecewise formula in (6):
            Δ_k = |Acc_k − Acc_global|   if F_k is undefined
                  |F_global − F_k|        otherwise

      Step 3 — ClientWeightedModelUpdate:
        Server updates aggregation weights with the subtraction rule (6):
            ω̄_k^t = ω̄_k^{t-1} − β · (Δ_k − (1/K)·Σ_i Δ_i)
        weights are clipped to [0, ∞) and re-normalised before aggregation.
        θ^{t+1} = (Σ ω_k^t · θ_k^t) / (Σ ω_k^t)

    Initialization:
        ω_k^0 = n_k / Σ n_i   (sample-count weights)
        Aggregate dataset statistics  S = { Pr(A=0,Y=1), Pr(A=1,Y=1) }
        from all clients (SecAgg) to use as the denominators in Formula (7).
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # β in Formula (6) — controls how aggressively weights are adjusted
        self.fairness_beta = getattr(args, "fairness_lambda", 0.1)

        self.set_slow_clients()
        self.set_clients(clientFairFed)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Fairness beta (λ): {self.fairness_beta}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.rs_eod = []
        self.rs_acc_gap = []
        self.rs_acc_std = []
        self.rs_acc_worst = []

        # ── Initialize ω_k^0 = n_k / Σ n_i  (Algorithm 1, line 1) ──────────
        # omega_bar: 未归一化权重 ω̄_k，是 Formula (6) 更新规则的操作对象
        # omega:     归一化权重 ω_k = ω̄_k / Σ_i ω̄_i，用于模型聚合
        total_train = sum(c.train_samples for c in self.clients)
        self.omega_bar = {c.id: c.train_samples / total_train for c in self.clients}
        self.omega = {c.id: c.train_samples / total_train for c in self.clients}

        # ── Dataset statistics S (Algorithm 1, line 2) ───────────────────────
        # SecAgg: aggregate Pr(A=0,Y=1) and Pr(A=1,Y=1) across ALL clients.
        # These are global denominators for Formula (7) and remain fixed.
        self._global_n_a0_y1, self._global_n_a1_y1, self._global_n = (
            self._aggregate_dataset_stats()
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization helper: collect Pr(A=0,Y=1) and Pr(A=1,Y=1) from clients
    # ─────────────────────────────────────────────────────────────────────────
    def _aggregate_dataset_stats(self):
        """
        Simulate SecAgg of dataset statistics S over all clients.
        Returns (global_n_a0_y1, global_n_a1_y1, global_n).
        """
        total_n = 0
        total_n_a0_y1 = 0
        total_n_a1_y1 = 0
        for client in self.clients:
            # 仅统计样本数量，不运行模型推理（避免对随机初始化模型的无效评估）
            stats = client.get_dataset_stats()
            total_n += stats["n"]
            total_n_a0_y1 += stats["n_a0_y1"]
            total_n_a1_y1 += stats["n_a1_y1"]
        print(
            f"[FairFed Init] Global n={total_n}, "
            f"n(A=0,Y=1)={total_n_a0_y1}, n(A=1,Y=1)={total_n_a1_y1}"
        )
        return total_n_a0_y1, total_n_a1_y1, total_n

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # ── Step 1: ClientLocalMetrics(k, θ^{t-1}) ───────────────────────
            # Each client evaluates the received global model on its local
            # training data and returns per-group TPR counts.
            metrics = {}
            for client in self.selected_clients:
                metrics[client.id] = client.compute_local_metrics()

            # Formula (7): F_global = Σ_k (n_k/n) · m_{global,k}
            # where m_{global,k} =   tpr_a0_k · Pr(A=0,Y=1|C=k) / Pr(Y=1,A=0)
            #                      − tpr_a1_k · Pr(A=1,Y=1|C=k) / Pr(Y=1,A=1)
            f_global, acc_global = self._compute_global_metrics(metrics)

            # ── Step 2: ClientMetricGap ───────────────────────────────────────
            # Broadcast F_global and Acc_global; each client computes Δ_k.
            deltas = {}
            for client in self.selected_clients:
                deltas[client.id] = client.compute_metric_gap(f_global, acc_global)

            mean_delta = float(np.mean(list(deltas.values())))  # (1/K) Σ_i Δ_i

            # ── 周期性评估（全局模型 θ^{t-1}，必须在本地训练之前）──────────────
            # 此时所有客户端的 model 均为 send_models() 下发的全局模型，评估结果一致。
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print(
                    f"F_global (EOD): {f_global:.4f} | Acc_global: {acc_global:.4f} | mean_Δ: {mean_delta:.4f}"
                )
                print("\nEvaluate global model with fairness metrics")
                self.evaluate()

            # ── Step 3: ClientWeightedModelUpdate ────────────────────────────
            # Formula (6): ω̄_k^t = ω̄_k^{t-1} − β·(Δ_k − (1/K)·Σ_i Δ_i)，截断至 ≥ 0
            # 注意：更新操作的基数是未归一化的 ω̄_k^{t-1}，而非归一化后的 ω_k。
            for client in self.selected_clients:
                delta_k = deltas[client.id]
                new_bar = self.omega_bar[client.id] - self.fairness_beta * (
                    delta_k - mean_delta
                )
                self.omega_bar[client.id] = max(0.0, new_bar)

            # 对全部 K 个客户端归一化：ω_k^t = ω̄_k^t / Σ_{i=1}^K ω̄_i^t
            # 未被选中的客户端保持 ω̄_k 不变，仅参与归一化分母的计算。
            total_bar = sum(self.omega_bar[c.id] for c in self.clients) or 1.0
            for c in self.clients:
                self.omega[c.id] = self.omega_bar[c.id] / total_bar

            # 为选中客户端设置聚合权重，然后开始本地训练
            for client in self.selected_clients:
                client.adjusted_weight = self.omega[client.id]
                client.train()

            # ── Receive models and aggregate with ω_k weights ─────────────────
            self._receive_models_fairfed()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()  # θ^{t+1} = Σ(ω̄_k θ_k) / Σ ω̄_k

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFairFed)
            print("\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    # ─────────────────────────────────────────────────────────────────────────
    # Formula (7): compute F_global and Acc_global from per-client statistics
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_global_metrics(self, metrics):
        """
        Formula (7):
          F_global = Σ_k (n_k/n) · m_{global,k}
          where
            m_{global,k} =   tpr_a0_k · [Pr(A=0,Y=1|C=k) / Pr(Y=1,A=0)]
                           − tpr_a1_k · [Pr(A=1,Y=1|C=k) / Pr(Y=1,A=1)]

        Pr(Y=1,A=a) = global_n_a_y1 / global_n   (from initialisation SecAgg)
        Pr(A=a,Y=1|C=k) = n_a_y1_k / n_k

        Acc_global = Σ_k (n_k/n) · acc_k  (standard weighted accuracy)
        """
        global_n = sum(m["n"] for m in metrics.values()) or 1
        p_a0_y1 = self._global_n_a0_y1 / self._global_n if self._global_n > 0 else 1e-9
        p_a1_y1 = self._global_n_a1_y1 / self._global_n if self._global_n > 0 else 1e-9

        f_global = 0.0
        acc_global = 0.0

        for cid, m in metrics.items():
            n_k = m["n"]
            if n_k == 0:
                continue
            w_k = n_k / global_n  # n_k / n

            # Accuracy contribution
            acc_global += w_k * (m["correct"] / n_k)

            # F_k contribution (only when both TPRs are defined)
            tpr_a0 = m["tpr_a0"]
            tpr_a1 = m["tpr_a1"]
            if tpr_a0 is not None and tpr_a1 is not None:
                pr_a0_y1_given_ck = m["n_a0_y1"] / n_k
                pr_a1_y1_given_ck = m["n_a1_y1"] / n_k
                m_global_k = tpr_a0 * pr_a0_y1_given_ck / (
                    p_a0_y1 + 1e-9
                ) - tpr_a1 * pr_a1_y1_given_ck / (p_a1_y1 + 1e-9)
                f_global += w_k * m_global_k

        return f_global, acc_global

    # ─────────────────────────────────────────────────────────────────────────
    # Receive uploaded models and set fairness-adjusted weights for aggregation
    # ─────────────────────────────────────────────────────────────────────────
    def _receive_models_fairfed(self):
        assert len(self.selected_clients) > 0
        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        total_w = 0.0

        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"]
                    / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"]
                    / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0.0

            if client_time_cost <= self.time_threthold:
                w = (
                    client.adjusted_weight
                    if client.adjusted_weight is not None
                    else self.omega.get(client.id, 1.0)
                )
                total_w += w
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(w)
                self.uploaded_models.append(client.model)

        for idx in range(len(self.uploaded_weights)):
            self.uploaded_weights[idx] /= total_w if total_w > 0 else 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate: 复用 serverbase.evaluate() 处理 acc/loss，再附加公平性指标
    # ─────────────────────────────────────────────────────────────────────────
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

        # EOD: 全局聚合所有客户端的 TP/正类计数后一次性计算 TPR 差。
        # 等价于在全局 pooled 测试集上评估，且保留符号（论文公式 2）：
        #   EOD = Pr(Ŷ=1|A=0,Y=1) - Pr(Ŷ=1|A=1,Y=1)
        # 不对 per-client EOD 做加权平均——无正类的大客户端（eod=0）
        # 权重大，会把整体 EOD 错误地拉向 0。
        total_tp_g0 = sum(s.get("n_tp_g0", 0) for s in stats)
        total_y1_g0 = sum(s.get("n_y1_g0", 0) for s in stats)
        total_tp_g1 = sum(s.get("n_tp_g1", 0) for s in stats)
        total_y1_g1 = sum(s.get("n_y1_g1", 0) for s in stats)
        tpr_g0 = total_tp_g0 / total_y1_g0 if total_y1_g0 > 0 else 0.0
        tpr_g1 = total_tp_g1 / total_y1_g1 if total_y1_g1 > 0 else 0.0
        eod = tpr_g0 - tpr_g1

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

        self.rs_eod.append(eod)
        self.rs_acc_gap.append(acc_gap)
        self.rs_acc_std.append(acc_std)
        self.rs_acc_worst.append(acc_worst)

        print(
            f"  [Fairness] EOD: {eod:.4f} | AccGap: {acc_gap:.4f} | "
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
