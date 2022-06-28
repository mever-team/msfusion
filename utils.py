import os
import torch
from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix,roc_auc_score,roc_curve,auc,f1_score,jaccard_score
import numpy as np
# from pytorch_lightning import metrics
import torchmetrics
import math

def device_check(ids:list):
    if torch.cuda.is_available():
        devices = []
        for i in ids:
            assert i < torch.cuda.device_count()
            devices.append(torch.device('cuda:'+str(i)))
        return devices[0],ids                  
        torch.device('cuda:'+str(gpu_id) if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception('No cuda available')
        
class Metrics():
    
    def __init__(self, metrics=['auc','iou','precision','recall','f1','accuracy'], num_classes=2, threshold=0.5, average=None, return_on_update=True):
        
        self.SUPPORTED_METRICS = {'auc':torchmetrics.functional.classification.auc,
                                  'iou': self.calculate_iou,
                                  'precision': self.calculate_precision,
                                  'recall': self.calculate_recall,
                                  'f1': self.calculate_f1,
                                  'accuracy': self.calculate_accuracy
                                 }
        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self.return_on_update = return_on_update
        self.init_metric_functions(metrics)
        
    def get_metric_names(self):
        return list(self.metric_functions.keys())
    
    def calculate_accuracy(self, conf_matrix:torch.tensor, average=None) -> torch.tensor:
        return conf_matrix.diag().sum() / conf_matrix.sum()

    def calculate_iou(self,conf_matrix:torch.tensor, average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_positive = conf_matrix.sum(0) - true_positive
        false_negative = conf_matrix.sum(1) - true_positive
        iou = true_positive / (true_positive + false_positive + false_negative)
        if average== 'macro':
            return iou.mean()
        else:
            return iou

    def calculate_precision(self,conf_matrix:torch.tensor, average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_positive = conf_matrix.sum(0) - true_positive
        precision  = true_positive / (true_positive + false_positive)
        if average== 'macro':
            return precision.mean()
        else:
            return precision

    def calculate_recall(self,conf_matrix:torch.tensor, average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_negative = conf_matrix.sum(1) - true_positive
        recall  = true_positive / (true_positive + false_negative)
        if average== 'macro':
            return recall.mean()
        else:
            return recall

    def calculate_f1(self,conf_matrix:torch.tensor,average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_negative = conf_matrix.sum(1) - true_positive
        false_positive = conf_matrix.sum(0) - true_positive
        precision  = true_positive / (true_positive + false_positive)
        recall  = true_positive / (true_positive + false_negative)
        f1 =  2*precision*recall/(precision+recall)
        if average== 'macro':
            return f1.mean()
        else:
            return f1
        
    def init_metric_functions(self,metrics):
        
        self.metric_functions = {}
        for metric in metrics:
            if metric in self.SUPPORTED_METRICS:
                self.metric_functions[metric] = self.SUPPORTED_METRICS[metric]
            
        self.confusion_matrix = torchmetrics.ConfusionMatrix(self.num_classes,threshold=self.threshold,compute_on_step=True)
        if 'auc' in self.metric_functions:
            self.roc = torchmetrics.ROC(1,compute_on_step=True) # in order to work for binary case roc needs num_classes ==1 
        
    def update(self,preds, targets):
        targets = targets.long()
        if len(self.metric_functions)>0:
            conf_matrix = self.confusion_matrix(preds, targets)
        if 'auc' in self.metric_functions:
            tp,fp,th = self.roc(preds, targets)
        res =  {metric:(self.metric_functions[metric](conf_matrix,average=self.average).numpy() if metric != 'auc' else self.metric_functions[metric](tp,fp).numpy()) for metric in self.metric_functions }
        return res
    
    def compute(self):
        if len(self.metric_functions)>0:
            conf_matrix = self.confusion_matrix.compute()
        if 'auc' in self.metric_functions:
            tp,fp,th = self.roc.compute()
        res =  {metric:(self.metric_functions[metric](conf_matrix,average=self.average).numpy() if metric != 'auc' else self.metric_functions[metric](tp,fp, reorder=True).numpy()) for metric in self.metric_functions }
        return res
    
    def reset(self):
        if len(self.metric_functions)>0:
            self.confusion_matrix.reset()
        if 'auc' in self.metric_functions:
            self.roc.reset()

class Binary_Metrics():
    
    def __init__(self, num_classes=2, threshold=0.5, average=None, return_on_update=True):
        
        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self.return_on_update = return_on_update
        self.init_metric_functions()
        self.init_agg_metrics()
        
    def init_agg_metrics(self):   
        self.aggregated_metrics = {'auc':[],
                          'iou': [],
                          'precision':[],
                          'recall':[],
                          'f1':[],
                          'accuracy': [],
                         }
        
    def get_metric_names(self):
        return list(self.conf_f.keys()) + list(self.auc_f.keys())
    
    def calculate_accuracy(self, conf_matrix:torch.tensor, average=None) -> torch.tensor:
        return conf_matrix.diag().sum() / conf_matrix.sum()

    def calculate_iou(self,conf_matrix:torch.tensor, average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_positive = conf_matrix.sum(0) - true_positive
        false_negative = conf_matrix.sum(1) - true_positive
        iou = true_positive / (true_positive + false_positive + false_negative)
        if average== 'macro':
            return iou.mean()
        else:
            return iou

    def calculate_precision(self,conf_matrix:torch.tensor, average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_positive = conf_matrix.sum(0) - true_positive
        precision  = true_positive / (true_positive + false_positive)
        if average== 'macro':
            return precision.mean()
        else:
            return precision

    def calculate_recall(self,conf_matrix:torch.tensor, average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_negative = conf_matrix.sum(1) - true_positive
        recall  = true_positive / (true_positive + false_negative)
        if average== 'macro':
            return recall.mean()
        else:
            return recall

    def calculate_f1(self,conf_matrix:torch.tensor,average=None) -> torch.tensor:
        true_positive = torch.diag(conf_matrix)
        false_negative = conf_matrix.sum(1) - true_positive
        false_positive = conf_matrix.sum(0) - true_positive
        precision  = true_positive / (true_positive + false_positive)
        recall  = true_positive / (true_positive + false_negative)
        f1 =  2*precision*recall/(precision+recall)
        if average== 'macro':
            return f1.mean()
        else:
            return f1
        
    def init_metric_functions(self):
        
        self.auc_f = {'auc':torchmetrics.functional.auc}
        self.conf_f = {
                          'iou': self.calculate_iou,
                          'precision': self.calculate_precision,
                          'recall': self.calculate_recall,
                          'f1': self.calculate_f1,
                          'accuracy': self.calculate_accuracy
                         }
        
        self.confusion_matrix = torchmetrics.ConfusionMatrix(self.num_classes,threshold=self.threshold, compute_on_step=False)
        self.roc = torchmetrics.ROC(1,compute_on_step=False) # in order to work for binary case roc needs num_classes ==1 
        
    def update(self,preds, targets):
        self.confusion_matrix(preds, targets)
        try:
            self.roc(preds, targets)
        except Exception as e:
            pass
        conf_matrix = self.confusion_matrix.compute()
#         print(conf_matrix)
        tp,fp,th  = self.roc.compute()
        conf_res =  {metric:self.conf_f[metric](conf_matrix,average=self.average).numpy()  for metric in self.conf_f }
        try:
            auc_res =  {metric:self.auc_f[metric](tp,fp).numpy() for metric in self.auc_f }
        except Exception as e:
            auc_res =  {metric:self.auc_f[metric](tp,fp, reorder=True).numpy() for metric in self.auc_f }
#             auc_res =  {metric:0.0 for metric in self.auc_f }
        res =   {**conf_res, **auc_res}
        
        for metric in self.aggregated_metrics:
            if self.average == 'macro':
                if math.isnan(conf_res['f1']):
                    continue
                self.aggregated_metrics[metric].append(res[metric])
            else:
                if math.isnan(conf_res['f1'][1]):
                    continue
                self.aggregated_metrics[metric].append(res[metric])
        return res
    
    def compute(self):
#         conf_matrix = self.confusion_matrix.compute()
#         tp,fp,th = self.roc.compute()
#         conf_res =  {metric:self.conf_f[metric](conf_matrix,average=self.average).numpy() for metric in self.conf_f }
#         try:
#             auc_res =  {metric:self.auc_f[metric](tp,fp).numpy() for metric in self.auc_f }
#         except Exception as e:
#             auc_res =  {metric:0.0 for metric in self.auc_f }
#         return  {**conf_res, **auc_res}
        res = {metric:np.stack(self.aggregated_metrics[metric]).mean(axis=0) for metric in self.aggregated_metrics}
        self.init_agg_metrics()
        return res
            
    
    
    
    
class Binary_Sklearn_Metrics():
    
    def __init__(self, metric_names=None, threshold=0.5):
        self.init_metric_functions(metric_names)
        self.threshold = threshold
        self.init_metrics()
        
    def get_metric_names(self):
        return list(self.metric_functions.keys())
    
    def custom_auc(self,gt, pred):
        fpr, tpr, thresholds  = roc_curve(gt, pred, drop_intermediate =True)
        return auc(fpr,tpr)
        
    def init_metric_functions(self,metric_names):
        
        metrics = {'accuracy':accuracy_score,
                  'precision':precision_score,
                  'recall':recall_score,
                  'f1':f1_score,
                  'auc':self.custom_auc,
                  #'confusion_matrix':confusion_matrix,
                  'iou':jaccard_score}
        if metric_names:
            self.metric_functions = {}
            for metric_name in metric_names:
                if metric_name in metrics:
                    self.metric_functions[metric_name] = metrics[metric_name]
        else:
            self.metric_functions = metrics
            
    def init_metrics(self):
        self.metrics = {metric:[] for metric in self.get_metric_names()}
        
    def aggregate_metrics(self):
        return {metric:np.mean(self.metrics[metric]) for metric in self.metrics}
                
    def compute(self, gt, pred, threshold=None,average=None):
        if not threshold:
            threshold = self.threshold
        result ={}
        for metric in self.metrics:
            if metric != 'auc':
                pred_ =  np.greater_equal(pred,threshold)
            else:
                pred_ = pred
            if metric in ['recall','precision','f1']:
                result[metric] = self.metric_functions[metric](gt, pred_,average=average)
            else:
                result[metric] = self.metric_functions[metric](gt, pred_)
            self.metrics[metric].append(result[metric])
        return result
    
    
def plot_roc_curve(gt, pred):
    fpr, tpr, thresholds  = roc_curve(gt, pred,drop_intermediate =False)
    roc_auc = auc(fpr,tpr)
    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(loc="lower right")