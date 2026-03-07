import time
import numpy as np
from flcore.clients.clientfairfed import clientFairFed
from flcore.servers.serverbase import Server
from threading import Thread


class FairFed(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Fairness parameters
        self.fairness_lambda = args.fairness_lambda if hasattr(args, 'fairness_lambda') else 0.1
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFairFed)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Fairness lambda: {self.fairness_lambda}")
        print("Finished creating server and clients.")

        self.Budget = []
        
        # Track fairness metrics over time
        self.fairness_history = {
            'demographic_parity': [],
            'average_accuracy': []
        }

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with fairness metrics")
                self.evaluate_with_fairness()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        
        # Print fairness summary
        print("\n" + "="*50)
        print("Fairness Summary")
        print("="*50)
        if len(self.fairness_history['demographic_parity']) > 0:
            avg_dp = np.mean(self.fairness_history['demographic_parity'])
            print(f"Average Demographic Parity Difference: {avg_dp:.4f}")
            print(f"Final Demographic Parity Difference: {self.fairness_history['demographic_parity'][-1]:.4f}")

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFairFed)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def evaluate_with_fairness(self):
        """
        Evaluate model performance with fairness metrics
        """
        stats = self.test_metrics()
        
        # Standard metrics
        stats_train = self.train_metrics()
        
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        
        # Fairness metrics aggregation
        all_fairness_metrics = stats[4] if len(stats) > 4 else []
        
        if len(all_fairness_metrics) > 0:
            # Compute average fairness metrics across clients
            total_samples = sum(stats[1])
            weighted_dp = 0
            
            for i, client_metrics in enumerate(all_fairness_metrics):
                if isinstance(client_metrics, dict) and 'demographic_parity' in client_metrics:
                    weight = stats[1][i] / total_samples
                    weighted_dp += client_metrics['demographic_parity'] * weight
            
            self.fairness_history['demographic_parity'].append(weighted_dp)
            self.fairness_history['average_accuracy'].append(test_acc)
            
            print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Train Loss: {train_loss:.4f}")
            print(f"Demographic Parity Difference: {weighted_dp:.4f}")
        else:
            print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Train Loss: {train_loss:.4f}")
            print("Fairness metrics not available (may require sensitive attribute configuration)")
    
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
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            tot_auc.append(auc*ns)
            fairness_metrics_list.append(fairness_metrics)
        
        ids = [c.id for c in self.clients]
        
        return ids, num_samples, tot_correct, tot_auc, fairness_metrics_list
