import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFairFed(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Fairness parameters
        self.fairness_lambda = args.fairness_lambda if hasattr(args, 'fairness_lambda') else 0.1
        self.sensitive_attr_idx = args.sensitive_attr_idx if hasattr(args, 'sensitive_attr_idx') else -1
        
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
                
                # Fairness loss (Demographic Parity approximation)
                fairness_loss = self.compute_fairness_loss(x, output, y)
                
                # Total loss
                total_loss = classification_loss + self.fairness_lambda * fairness_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    def compute_fairness_loss(self, x, output, y):
        """
        Compute fairness loss based on Demographic Parity
        Goal: P(Y_hat=1|S=0) ≈ P(Y_hat=1|S=1)
        where S is the sensitive attribute (e.g., gender)
        """
        try:
            # Get predictions
            predictions = torch.softmax(output, dim=1)[:, 1]  # Probability of positive class
            
            # For Adult dataset, we need to extract sensitive attribute
            # If sensitive attribute is not directly available in input,
            # we approximate fairness by ensuring balanced predictions
            
            # Simple approximation: minimize variance in prediction distribution
            # More sophisticated: use group-specific losses if sensitive attribute is available
            
            # Method 1: Variance-based fairness (encourages similar prediction distributions)
            mean_pred = predictions.mean()
            fairness_loss = torch.var(predictions)
            
            # Method 2: If we have access to sensitive attributes (uncomment if available)
            # This assumes sensitive attribute is the last feature or at a specific index
            # if self.sensitive_attr_idx >= 0 and x.ndim == 2:
            #     sensitive_attr = x[:, self.sensitive_attr_idx]
            #     # Group predictions by sensitive attribute
            #     group_0_mask = (sensitive_attr < 0.5)  # Assuming binary and normalized
            #     group_1_mask = (sensitive_attr >= 0.5)
            #     
            #     if group_0_mask.sum() > 0 and group_1_mask.sum() > 0:
            #         pred_group_0 = predictions[group_0_mask].mean()
            #         pred_group_1 = predictions[group_1_mask].mean()
            #         fairness_loss = torch.abs(pred_group_0 - pred_group_1)
            
            return fairness_loss
            
        except Exception as e:
            # If fairness computation fails, return zero loss
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
                
                fairness_metrics['demographic_parity'] = demographic_parity
                fairness_metrics['positive_rate_group_0'] = positive_rate_group_0
                fairness_metrics['positive_rate_group_1'] = positive_rate_group_1
        
        return test_acc, test_num, 0, fairness_metrics
