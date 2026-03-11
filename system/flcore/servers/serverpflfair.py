from flcore.clients.clientpflfair import clientPFLFair
from flcore.servers.serverfairfed import FairFed
from flcore.servers.serverbase import Server


class PFLFair(FairFed):
    """
    PFL-Fair: Personalized Federated Learning with Fairness.

    = FairFed (公平加权全局聚合) + FedALA (自适应本地聚合)

    算法流程（每轮）：
      1. send_models():
         服务器向所有客户端发送当前全局模型；
         客户端通过 ALA 将公平全局模型与本地模型自适应聚合（个性化适配）。

      2. [FairFed Step 1] compute_local_metrics():
         客户端在聚合后的本地模型上评估本地 TPR，上报至服务器。

      3. [FairFed Step 2] compute_metric_gap():
         服务器广播 F_global 和 Acc_global，客户端计算公平差距 Δ_k。

      4. [FairFed Step 3] client.train():
         客户端以调整后的权重 ω_k 为聚合权重，进行本地 SGD 训练。

      5. aggregate_parameters():
         服务器以公平权重 ω_k 加权聚合各客户端模型，得到下一轮全局模型。

    与 FairFed 的**唯一差异**：
      - 客户端类：clientFairFed → clientPFLFair（增加 ALA 初始化）
      - send_models()：直接参数覆盖 → ALA 自适应聚合
      - 其余逻辑（train loop、_evaluate_fairness、save_results）完全继承
    """

    def __init__(self, args, times):
        # 直接调用 Server.__init__ 绕过 FairFed 中的 set_clients(clientFairFed)，
        # 然后手动完成 FairFed 相同的初始化逻辑，仅将客户端类换为 clientPFLFair。
        Server.__init__(self, args, times)

        self.fairness_beta = getattr(args, "fairness_lambda", 0.1)

        self.set_slow_clients()
        self.set_clients(clientPFLFair)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Fairness beta (λ): {self.fairness_beta}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.rs_eod = []
        self.rs_acc_gap = []
        self.rs_acc_std = []
        self.rs_acc_worst = []

        # ω_k^0 = n_k / Σ n_i
        total_train = sum(c.train_samples for c in self.clients)
        self.omega_bar = {c.id: c.train_samples / total_train for c in self.clients}
        self.omega = {c.id: c.train_samples / total_train for c in self.clients}

        # SecAgg dataset statistics S（Algorithm 1, line 2）
        self._global_n_a0_y1, self._global_n_a1_y1, self._global_n = (
            self._aggregate_dataset_stats()
        )

    def send_models(self):
        """
        PFL-Fair 的 send_models 分两步：

          1. 在覆盖客户端模型之前，先把「上一轮训练完的本地模型」克隆保存。
             ALA 需要 (global, local_prev) 两个不同的模型来做自适应混合。
             如果用 set_parameters 覆盖后再做 ALA，两者参数完全相同，
             ALA 内部 sum(diff)==0 判断会直接 return，等同于没有 ALA。

          2. 再将全局模型参数复制到客户端（供 compute_local_metrics 正确
             评估纯全局模型的公平性指标，而非被 ALA 偏移后的个性化模型）。
        """
        import copy
        import time as _time

        assert len(self.clients) > 0
        for client in self.clients:
            start_time = _time.time()
            # ① 先保存上一轮训练后的本地模型（ALA 的 local 侧输入）
            client._local_model_before_ala = copy.deepcopy(client.model)
            # ② 用全局模型覆盖客户端模型（供 compute_local_metrics 正确评估）
            client.set_parameters(self.global_model)
            # ③ 同时保存全局模型引用（ALA 的 global 侧输入），只读不改
            client._global_model_snapshot = self.global_model
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (_time.time() - start_time)
