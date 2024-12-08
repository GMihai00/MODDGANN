from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
from typing import List

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import logging

class ClassificationPerformanceMetrics:
    
    def __init__(self, y=None, y_pred=None, accuracy=None, precision=None, recall=None, f1=None, roc=None, roc_auc=None):
        # If y and y_pred are provided, calculate the metrics
        
        if y is not None and y_pred is not None:
            y_pred_one_hot = np.zeros_like(y_pred)
            y_pred_one_hot[np.arange(len(y_pred)), y_pred.argmax(axis=1)] = 1
            self.accuracy = accuracy_score(y, y_pred_one_hot)
            self.precision = precision_score(y, y_pred_one_hot, average='weighted')  
            self.recall = recall_score(y, y_pred_one_hot, average='weighted')
            self.f1_score = f1_score(y, y_pred_one_hot, average='weighted')
            self.roc_auc_score = roc_auc_score(y, y_pred_one_hot)
            
            for i in range(y.shape[1]):
                self.roc_curve = []
                self.roc_curve.append(roc_curve(y[:, i], y_pred_one_hot[:, i]))

        # If pre-computed metrics are provided, use them directly
        elif accuracy is not None and precision is not None and recall is not None and f1_score is not None and roc_curve is not None and roc_auc_score is not None:
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1_score = f1
            self.roc_curve = roc
            self.roc_auc_score = roc_auc
        else:
            raise ValueError("Either 'y' and 'y_pred' must be provided or all metrics must be provided.")

    def plot_roc_curve(self):
        if self.roc_curve is None:
            print("ROC curve cannot be plotted. Ensure input data is binary and valid.")
            return
        
        for  fpr, tpr, _ in self.roc_curve:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {self.roc_auc_score:.4f})", color='blue', lw=2)
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)  # Diagonal line
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.show()

    def __str__(self):
        result = (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1 Score: {self.f1_score:.4f}\n"
        )
        if self.roc_auc_score is not None:
            result += f"ROC AUC Score: {self.roc_auc_score:.4f}\n"
        else:
            result += "ROC AUC Score: Not applicable (check input data)\n"
        return result
        
        
def calculate_metrics_average(metrics_list: List[ClassificationPerformanceMetrics]) -> ClassificationPerformanceMetrics:

    accuracy_mean = np.mean([metric.accuracy for metric in metrics_list])
    precision_mean = np.mean([metric.precision for metric in metrics_list])
    recall_mean = np.mean([metric.recall for metric in metrics_list])
    f1_score_mean = np.mean([metric.f1_score for metric in metrics_list])
    roc_auc_score_mean = np.mean([metric.roc_auc_score for metric in metrics_list])
    
    roc_curve_mean = []
    
    for it in range(0, len(metrics_list[0].roc_curve)):
        tpr_values = []
        fpr_values = []
        
        for metric in metrics_list:
            tpr, fpr, _  = metric.roc_curve[it]

            tpr_values.append(tpr)
            fpr_values.append(fpr)
            
        roc_curve_mean.append((np.mean(fpr_values, axis=0), np.mean(tpr_values, axis=0), None))
    
    return ClassificationPerformanceMetrics(
        accuracy=accuracy_mean,
        precision=precision_mean,
        recall=recall_mean,
        f1=f1_score_mean,
        roc_auc=roc_auc_score_mean,
        roc=roc_curve_mean
    )
    
def display_training_results(training_results: List[ClassificationPerformanceMetrics]):
    
    for it, metric in enumerate(training_results):
        logging.info(f"Results for iteration {it}: \n {str(metric)}")
    
    average_metrics = calculate_metrics_average(training_results)
    
    logging.info(f"Average training results:\n {str(average_metrics)}")
    
    average_metrics.plot_roc_curve()
