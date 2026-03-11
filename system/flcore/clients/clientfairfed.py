import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFairFed(Client):
    """
    FairFed client implementing Algorithm 1 (tracking EOD).

    State variables:
      local_tpr_a0, local_tpr_a1 : Pr(Ŷ=1 | A=0/1, Y=1, C=k)  — for Formula (7)
      local_acc                   : local accuracy on training data
      local_F_k                   : local EOD contribution term m_{global,k}
                                    (None if no positive samples for one group)
      delta_k                     : Δ_k computed in Step 2
      adjusted_weight             : ω̄_k^t set by server in Step 3
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.sensitive_attr_idx = getattr(args, "sensitive_attr_idx", -1)

        # per-round state
        self.local_tpr_a0 = None  # Pr(Ŷ=1 | A=0, Y=1, C=k)
        self.local_tpr_a1 = None  # Pr(Ŷ=1 | A=1, Y=1, C=k)
        self.n_a0_y1 = 0  # local count of (A=0, Y=1) samples
        self.n_a1_y1 = 0  # local count of (A=1, Y=1) samples
        self.local_acc = 0.0
        self.local_F_k = None  # None means F_k undefined for this client
        self.delta_k = 0.0
        self.adjusted_weight = None  # ω̄_k^t
        self._local_n_total = (
            0  # total training samples from last compute_local_metrics
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset statistics helper (Algorithm 1, line 2: SecAgg of S)
    # 仅统计样本数量，不运行模型推理。用于服务器初始化阶段聚合全局
    # Pr(A=0,Y=1) 和 Pr(A=1,Y=1)，避免对随机初始化模型的无效评估。
    # ─────────────────────────────────────────────────────────────────────────
    def get_dataset_stats(self):
        """返回 { n, n_a0_y1, n_a1_y1 }，纯样本计数，不依赖模型输出。"""
        trainloader = self.load_train_data()
        n = 0
        n_a0_y1 = 0
        n_a1_y1 = 0
        with torch.no_grad():
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                n += y.shape[0]
                if self.sensitive_attr_idx >= 0 and x.ndim == 2:
                    a = (x[:, self.sensitive_attr_idx] >= 0).long()
                    pos = y == 1
                    n_a0_y1 += ((~a.bool()) & pos).sum().item()
                    n_a1_y1 += ((a.bool()) & pos).sum().item()
        return {"n": n, "n_a0_y1": n_a0_y1, "n_a1_y1": n_a1_y1}

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — ClientLocalMetrics(k, θ^{t-1})
    # Evaluate the *received global model* on local training data.
    # Returns the quantities needed by the server to compute F_global (Formula 7)
    # and Acc_global, together with local sample counts.
    #
    # Returned dict (one row of the SecAgg table):
    #   {
    #     "n"          : total local training samples,
    #     "correct"    : number of correctly classified samples,
    #     "tpr_a0"     : Pr(Ŷ=1 | A=0, Y=1, C=k)  — numerator term,
    #     "tpr_a1"     : Pr(Ŷ=1 | A=1, Y=1, C=k),
    #     "n_a0_y1"    : count of (A=0, Y=1) in this client,
    #     "n_a1_y1"    : count of (A=1, Y=1) in this client,
    #   }
    # ─────────────────────────────────────────────────────────────────────────
    def compute_local_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        correct = 0
        total = 0
        tp_a0 = 0  # true-positive count for A=0
        tp_a1 = 0
        cnt_a0_y1 = 0
        cnt_a1_y1 = 0

        with torch.no_grad():
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                preds = torch.argmax(self.model(x), dim=1)

                correct += (preds == y).sum().item()
                total += y.shape[0]

                if self.sensitive_attr_idx >= 0 and x.ndim == 2:
                    # Binarise sensitive attribute: A=1 if feature >= 0, else A=0
                    a = (x[:, self.sensitive_attr_idx] >= 0).long()
                    pos = y == 1

                    mask_a0_y1 = (~a.bool()) & pos
                    mask_a1_y1 = (a.bool()) & pos

                    tp_a0 += (preds[mask_a0_y1] == 1).sum().item()
                    tp_a1 += (preds[mask_a1_y1] == 1).sum().item()
                    cnt_a0_y1 += mask_a0_y1.sum().item()
                    cnt_a1_y1 += mask_a1_y1.sum().item()

        self.local_acc = correct / total if total > 0 else 0.0
        self.n_a0_y1 = cnt_a0_y1
        self.n_a1_y1 = cnt_a1_y1
        self._local_n_total = total  # stored for compute_metric_gap (Formula 7)

        # Pr(Ŷ=1 | A=0/1, Y=1, C=k) — set to None if no samples in that group
        self.local_tpr_a0 = tp_a0 / cnt_a0_y1 if cnt_a0_y1 > 0 else None
        self.local_tpr_a1 = tp_a1 / cnt_a1_y1 if cnt_a1_y1 > 0 else None

        return {
            "n": total,
            "correct": correct,
            "tpr_a0": self.local_tpr_a0,
            "tpr_a1": self.local_tpr_a1,
            "n_a0_y1": cnt_a0_y1,
            "n_a1_y1": cnt_a1_y1,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — ClientMetricGap(k, θ^{t-1}, F_global, Acc_global)
    #
    # Server broadcasts F_global and Acc_global; client computes:
    #
    #   F_k = local EOD = Pr(Ŷ=1|A=0,Y=1,C=k) − Pr(Ŷ=1|A=1,Y=1,C=k)
    #         (undefined if either TPR is None)
    #
    #   Δ_k = |Acc_k − Acc_global|               if F_k is undefined
    #         |F_global − F_k|                    otherwise
    #   (Formula 6 piecewise definition)
    #
    # Returns Δ_k so the server can compute mean_Δ = (1/K) Σ Δ_i.
    # ─────────────────────────────────────────────────────────────────────────
    def compute_metric_gap(self, f_global, acc_global):
        if self.local_tpr_a0 is None or self.local_tpr_a1 is None:
            # F_k undefined — use accuracy gap as fallback
            self.local_F_k = None
            self.delta_k = abs(self.local_acc - acc_global)
        else:
            # Simple per-client EOD (bounded in [-1, 1]).
            # Do NOT use Formula (7) normalized m_global_k here: for K=20 with
            # heavy heterogeneity, female-dominated clients get m_global_k ≈ +1.7
            # → Δ_k ≈ 2.1 → FairFed aggressively down-weights them even though
            # they HELP fairness. Simple F_k keeps Δ_k in [0, 2] and prevents
            # this counter-productive penalisation of "pro-fairness" clients.
            self.local_F_k = self.local_tpr_a0 - self.local_tpr_a1
            self.delta_k = abs(f_global - self.local_F_k)
        return self.delta_k

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()

        criterion_per_sample = torch.nn.CrossEntropyLoss(reduction="none")

        # Pre-compute Y=1 group counts from the FULL local dataset (stable).
        # EOD-targeted RW: only positive-class (Y=1) samples are reweighted.
        #
        # WHY Y=1 only (not all samples):
        # For a client with 17 females (y1_f=0) and 250 males, the old
        # all-sample RW up-weights those 17 female Y=0 samples by 8x, teaching
        # the model "female → Y=0" ever more strongly → TPR_female drops →
        # EOD becomes MORE negative. Y=1-only RW avoids this.
        w_y1_a0, w_y1_a1 = 1.0, 1.0
        if self.sensitive_attr_idx >= 0:
            n_y1_a0_total, n_y1_a1_total = 0, 0
            with torch.no_grad():
                for bx, by in trainloader:
                    bx = bx[0] if isinstance(bx, list) else bx
                    if bx.ndim == 2:
                        ba = bx[:, self.sensitive_attr_idx] >= 0
                        bpos = by == 1
                        n_y1_a0_total += int((bpos & ~ba).sum().item())
                        n_y1_a1_total += int((bpos & ba).sum().item())
            if n_y1_a0_total >= 3 and n_y1_a1_total >= 3:
                n_y1_total = n_y1_a0_total + n_y1_a1_total
                # Cap at 10x to limit gradient variance from very small groups
                w_y1_a0 = min(n_y1_total / (2.0 * n_y1_a0_total), 10.0)
                w_y1_a1 = min(n_y1_total / (2.0 * n_y1_a1_total), 10.0)
            # else: fall back to standard ERM (avoid harmful up-weighting of
            # clients where one group has zero or near-zero Y=1 samples)

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)

                if (
                    (w_y1_a0 != 1.0 or w_y1_a1 != 1.0)
                    and self.sensitive_attr_idx >= 0
                    and x.ndim == 2
                ):
                    a = x[:, self.sensitive_attr_idx] >= 0
                    pos = y == 1
                    weights = torch.ones(
                        x.shape[0], device=self.device, dtype=torch.float
                    )
                    weights[pos & ~a] = w_y1_a0
                    weights[pos & a] = w_y1_a1
                    loss = (criterion_per_sample(output, y) * weights).mean()
                else:
                    loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    # test_metrics_fairness() 由 clientbase.Client 提供，返回完整的
    # (correct, total, 0, {"eod", "n_correct_g0", "n_g0", "n_correct_g1", "n_g1"})
    # 此处不再覆写，保持与 FedAvg_Fair / FedALA_Fair / PFL-Fair 的评估逻辑一致。
