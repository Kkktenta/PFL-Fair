from flcore.clients.clientfairfed import clientFairFed
from utils.data_utils import read_client_data
from utils.ALA import ALA


class clientPFLFair(clientFairFed):
    """
    PFL-Fair client: FairFed + FedALA.

    继承 clientFairFed 的全部 FairFed Algorithm 1 流程：
      - get_dataset_stats()      — 初始化：SecAgg 数据集统计
      - compute_local_metrics()  — Step 1：计算本地 TPR
      - compute_metric_gap()     — Step 2：计算 Δ_k
      - train()                  — Step 3：本地 SGD 训练
      - test_metrics_fairness()  — 评估（继承自 clientbase，含完整公平性指标）

    在此基础上加入 FedALA 的自适应本地聚合（ALA）：
      服务器下发**公平加权**全局模型后，客户端不直接覆盖本地模型，而是
      通过 local_initialization() 在全局模型与当前本地模型之间寻找最优
      线性插值，再进行本地训练——实现个性化公平的双重目标。
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

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

    def local_initialization(self, received_global_model):
        """ALA：在公平全局模型与本地模型之间自适应聚合，代替直接参数覆盖。"""
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)

    def train(self):
        """
        在 FairFed 的本地 SGD 训练之前先执行 ALA 自适应聚合。

        正确流程：
          1. serverpflfair.send_models() 在覆盖本地模型之前已将「上一轮训练完的
             本地模型」存入 self._local_model_before_ala，并将全局模型引用存入
             self._global_model_snapshot。
          2. 此处先将 client.model 恢复为 local_prev，再调用 ALA 混合
             (global_snapshot, local_prev) → 得到个性化起点。
          3. 之后调用父类 clientFairFed.train() 从个性化起点做带公平重加权的 SGD。

        关键：ALA 需要 global != local 才能工作；若直接对覆盖后的全局模型做 ALA，
        两者参数相同 → ALA 内部 sum(diff)==0 → 立即 return → ALA 形同虚设。
        """
        local_prev = getattr(self, "_local_model_before_ala", None)
        snapshot = getattr(self, "_global_model_snapshot", None)

        if local_prev is not None and snapshot is not None:
            # 恢复上一轮的本地模型作为 ALA 的 local 侧
            self.set_parameters(local_prev)
            # ALA 混合：(global_snapshot, local_prev) → self.model
            self.ALA.adaptive_local_aggregation(snapshot, self.model)
            self._local_model_before_ala = None
            self._global_model_snapshot = None
        # 调用父类（clientFairFed）的带公平重加权 SGD 训练
        super().train()
