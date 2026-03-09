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
