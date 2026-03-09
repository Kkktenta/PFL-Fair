import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs["train_slow"]
        self.send_slow = kwargs["send_slow"]
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.sensitive_attr_idx = getattr(args, "sensitive_attr_idx", -1)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(
            self.dataset, self.id, is_train=True, few_shot=self.few_shot
        )
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(
            self.dataset, self.id, is_train=False, few_shot=self.few_shot
        )
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def test_metrics_fairness(self):
        """Test metrics with fairness evaluation (EOD + per-group accuracy).

        Returns:
            (correct, total, auc=0, fairness_metrics) where fairness_metrics
            contains 'eod', 'n_correct_g0', 'n_g0', 'n_correct_g1', 'n_g1'.
            Returns empty fairness_metrics if sensitive_attr_idx < 0.
        """
        if self.sensitive_attr_idx < 0:
            ct, ns, auc = self.test_metrics()
            return ct, ns, auc, {}

        testloader = self.load_test_data()
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_labels, all_sensitive = [], [], []

        with torch.no_grad():
            for x, y in testloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                preds = torch.argmax(self.model(x), dim=1)
                correct += (preds == y).sum().item()
                total += y.shape[0]
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                if x.ndim == 2:
                    all_sensitive.append(x[:, self.sensitive_attr_idx].cpu().numpy())

        if not all_sensitive:
            return correct, total, 0, {}

        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        sensitive_arr = (np.concatenate(all_sensitive) >= 0).astype(int)

        fairness_metrics = {}
        # 论文公式 (2): EOD = |Pr(Ŷ=1|A=0,Y=1) - Pr(Ŷ=1|A=1,Y=1)|
        # 只考虑正类样本（Equal Opportunity Difference）。
        # 返回各客户端的 TP 和正类计数，由服务端在所有客户端上全局聚合后再算 TPR 差，
        # 避免在单个客户端上做加权平均（无正类的大客户端会把 EOD 错误地拉向 0）。
        mask_y1_g0 = (sensitive_arr == 0) & (labels_arr == 1)
        mask_y1_g1 = (sensitive_arr == 1) & (labels_arr == 1)
        fairness_metrics["n_tp_g0"] = (
            int(preds_arr[mask_y1_g0].sum()) if mask_y1_g0.any() else 0
        )
        fairness_metrics["n_y1_g0"] = int(mask_y1_g0.sum())
        fairness_metrics["n_tp_g1"] = (
            int(preds_arr[mask_y1_g1].sum()) if mask_y1_g1.any() else 0
        )
        fairness_metrics["n_y1_g1"] = int(mask_y1_g1.sum())
        # 保留单客户端 EOD 供调试参考（服务端不用此字段计算全局 EOD）
        tpr_g0 = (
            fairness_metrics["n_tp_g0"] / fairness_metrics["n_y1_g0"]
            if fairness_metrics["n_y1_g0"] > 0
            else 0.0
        )
        tpr_g1 = (
            fairness_metrics["n_tp_g1"] / fairness_metrics["n_y1_g1"]
            if fairness_metrics["n_y1_g1"] > 0
            else 0.0
        )
        fairness_metrics["eod"] = abs(tpr_g0 - tpr_g1)

        mask_g0 = sensitive_arr == 0
        mask_g1 = sensitive_arr == 1
        fairness_metrics["n_correct_g0"] = (
            int((preds_arr[mask_g0] == labels_arr[mask_g0]).sum())
            if mask_g0.any()
            else 0
        )
        fairness_metrics["n_g0"] = int(mask_g0.sum())
        fairness_metrics["n_correct_g1"] = (
            int((preds_arr[mask_g1] == labels_arr[mask_g1]).sum())
            if mask_g1.any()
            else 0
        )
        fairness_metrics["n_g1"] = int(mask_g1.sum())

        return correct, total, 0, fairness_metrics

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(
            item,
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"),
        )

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(
            os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")
        )

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
