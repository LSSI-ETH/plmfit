import torch
import os
import json
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryMatthewsCorrCoef, BinaryConfusionMatrix, BinaryROC, MulticlassAccuracy, MulticlassMatthewsCorrCoef, MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score, SpearmanCorrCoef
from torchmetrics.text import Perplexity
from torchmetrics.classification import MultilabelAccuracy, MultilabelMatthewsCorrCoef, MultilabelConfusionMatrix, MultilabelExactMatch, MultilabelF1Score
from plmfit.shared_utils.utils import get_test_dataset

class BaseMetrics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.preds_list = []
        self.actual_list = []
        self.ids = []
        self.report = None

    def add(self, preds, actual, ids):
        """
        Default add method: appends predictions, actual values, and IDs.
        Override in subclasses if a different behavior is needed.
        """
        # For tasks where we expect scalar predictions per sample
        preds_list = preds.tolist()
        actual_list = actual.tolist()
        ids_list = ids.tolist()

        if len(preds_list) > 1:
            self.preds_list.extend(preds_list)
        else:
            self.preds_list.append(preds.item())

        if len(actual_list) > 1:
            self.actual_list.extend(actual_list)
        else:
            self.actual_list.append(actual.item())

        if len(ids_list) > 1:
            self.ids.extend(ids_list)
        else:
            self.ids.append(ids.item())

    def save_metrics(self, path):
        metrics_path = f"{path}_metrics.json"
        if self.report is None:
            self.get_metrics()
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if "pred_data" in existing_data:
                existing_data["pred_data"]["preds"].extend(
                    self.report["pred_data"]["preds"]
                )
                existing_data["pred_data"]["actual"].extend(
                    self.report["pred_data"]["actual"]
                )
                existing_data["pred_data"]["ids"].extend(
                    self.report["pred_data"]["ids"]
                )
                self.report = existing_data
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=4)


class ClassificationMetrics(BaseMetrics):
    def __init__(self, no_classes=1):
        super().__init__()
        self.no_classes = no_classes
        if self.no_classes < 2:
            # Binary classification metrics
            self.acc = BinaryAccuracy()
            self.roc_auc = BinaryAUROC()
            self.mcc = BinaryMatthewsCorrCoef()
            self.cm = BinaryConfusionMatrix()
            self.roc = BinaryROC()
        else:
            # Multiclass classification metrics
            self.acc = MulticlassAccuracy(num_classes=self.no_classes)
            self.micro_acc = MulticlassAccuracy(
                num_classes=self.no_classes, average="micro"
            )
            self.mcc = MulticlassMatthewsCorrCoef(num_classes=self.no_classes)
            self.cm = MulticlassConfusionMatrix(num_classes=self.no_classes)

    def calculate(self, preds, actual):
        self.acc.update(preds, actual)
        if self.no_classes < 2:
            self.roc_auc.update(preds, actual)
            self.roc.update(preds, actual.int())
        else:
            self.micro_acc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.cm.update(preds, actual)

    def get_metrics(self, device="cpu"):
        preds_tensor = torch.tensor(self.preds_list, device=device)
        actual_tensor = torch.tensor(self.actual_list, device=device)
        self.calculate(preds_tensor, actual_tensor)

        if self.no_classes < 2:
            fpr, tpr, thresholds = list(self.roc.compute())
            self.report = {
                "main": {
                    "accuracy": self.acc.compute().item(),
                    "roc_auc": self.roc_auc.compute().item(),
                    "mcc": self.mcc.compute().item(),
                    "confusion_matrix": self.cm.compute().tolist(),
                },
                "roc_auc_data": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "roc_auc_val": self.roc_auc.compute().item(),
                },
                "pred_data": {
                    "preds": self.preds_list,
                    "actual": self.actual_list,
                    "ids": self.ids,
                },
            }
        else:
            self.report = {
                "main": {
                    "accuracy": self.acc.compute().item(),
                    "micro_accuracy": self.micro_acc.compute().item(),
                    "mcc": self.mcc.compute().item(),
                    "confusion_matrix": self.cm.compute().tolist(),
                },
                "pred_data": {
                    "preds": self.preds_list,
                    "actual": self.actual_list,
                    "ids": self.ids,
                },
            }
        return self.report


class RegressionMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.mse = MeanSquaredError()
        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        self.spearman = SpearmanCorrCoef()

    def calculate(self, preds, actual):
        self.mse.update(preds, actual)
        self.rmse.update(preds, actual)
        self.mae.update(preds, actual)
        self.r2.update(preds, actual)
        self.spearman.update(preds, actual)

    def get_metrics(self, device="cpu"):
        preds_tensor = torch.tensor(self.preds_list, device=device)
        actual_tensor = torch.tensor(self.actual_list, device=device)
        self.calculate(preds_tensor, actual_tensor)

        metrics = {
            "mse": self.mse.compute().item(),
            "rmse": self.rmse.compute().item(),
            "mae": self.mae.compute().item(),
            "r_sq": self.r2.compute().item(),
            "spearman": self.spearman.compute().item(),
        }
        self.report = {
            "main": metrics,
            "pred_data": {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "ids": self.ids,
                "eval_metrics": metrics,
            },
        }
        return self.report


class MaskedLMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.perplexity = Perplexity(ignore_index=-100)

    def calculate(self, preds, actual):
        self.perplexity.update(preds, actual)

    def get_metrics(self, device="cpu"):
        preds_tensor = torch.tensor(self.preds_list, device=device)
        actual_tensor = torch.tensor(self.actual_list, device=device)
        self.calculate(preds_tensor, actual_tensor)
        metrics = {"perplexity": self.perplexity.compute().item()}
        self.report = {"main": metrics}
        return self.report


class TokenClassificationMetrics(BaseMetrics):
    def __init__(self, no_classes=1):
        super().__init__()
        self.no_classes = no_classes
        self.acc = MulticlassAccuracy(num_classes=self.no_classes, ignore_index=-100)
        self.micro_acc = MulticlassAccuracy(
            num_classes=self.no_classes, average="micro", ignore_index=-100
        )
        self.mcc = MulticlassMatthewsCorrCoef(
            num_classes=self.no_classes, ignore_index=-100
        )
        self.cm = MulticlassConfusionMatrix(
            num_classes=self.no_classes, ignore_index=-100
        )

    def add(self, preds, actual, ids):
        """
        Override add: For token classification we assume the inputs are already token‐wise lists.
        """
        self.preds_list.extend(preds.tolist())
        self.actual_list.extend(actual.tolist())
        ids_list = ids.tolist()
        if len(ids_list) > 1:
            self.ids.extend(ids_list)
        else:
            self.ids.append(ids.item())

    def calculate(self, preds, actual):
        self.acc.update(preds, actual)
        self.micro_acc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.cm.update(preds, actual)

    def get_metrics(self, device="cpu"):
        preds_tensor = torch.tensor(self.preds_list, device=device)
        actual_tensor = torch.tensor(self.actual_list, device=device)
        self.calculate(preds_tensor, actual_tensor)
        self.report = {
            "main": {
                "accuracy": self.acc.compute().item(),
                "micro_accuracy": self.micro_acc.compute().item(),
                "mcc": self.mcc.compute().item(),
                "confusion_matrix": self.cm.compute().tolist(),
            },
            "pred_data": {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "ids": self.ids,
            },
        }
        return self.report


class MultilabelClassificationMetrics(BaseMetrics):
    def __init__(self, no_labels=1):
        super().__init__()
        self.no_labels = no_labels
        self.macro_acc = MultilabelAccuracy(num_labels=self.no_labels, ignore_index=-100)
        self.micro_acc = MultilabelAccuracy(
            num_labels=self.no_labels, average="micro", ignore_index=-100
        )
        self.mcc = MultilabelMatthewsCorrCoef(
            num_labels=self.no_labels, ignore_index=-100
        )
        self.exact_match = MultilabelExactMatch(
            num_labels=self.no_labels, ignore_index=-100
        )
        self.macro_f1 = MultilabelF1Score(num_labels=self.no_labels, ignore_index=-100)
        self.micro_f1 = MultilabelF1Score(
            num_labels=self.no_labels, average="micro", ignore_index=-100
        )
        # Per-label metrics (for those metrics that support per-label evaluation)
        self.per_label_acc = MultilabelAccuracy(
            num_labels=self.no_labels, average=None, ignore_index=-100
        )
        self.per_label_f1 = MultilabelF1Score(
            num_labels=self.no_labels, average=None, ignore_index=-100
        )
        self.cm = MultilabelConfusionMatrix(
            num_labels=self.no_labels, ignore_index=-100
        )

    def add(self, preds, actual, ids, division=None):
        """
        Override add: Extends the stored lists with predictions, actuals, and ids.
        If 'division' is provided (as a string column name), then the dataset is queried
        to select only those rows where dataset[division] == 1. Only the corresponding
        predictions, actuals, and ids are added.
        
        Assumes a function `get_dataset()` exists and returns a pandas DataFrame
        that contains an 'id' column and a column with the name provided in division.
        """
        if division is not None:
            # Get the dataset reference (assumed to be available via a helper function)
            dataset = get_test_dataset()  
            # Get the set of valid IDs based on the division column condition (1/True)
            valid_ids = set(dataset.loc[dataset[division] == 1, 'id'].tolist())
            
            # Convert tensors to lists for easier filtering
            preds_list = preds.tolist()
            actual_list = actual.tolist()
            ids_list = ids.tolist()
            
            # Filter out predictions, actuals, and ids for valid ids only
            filtered_preds = []
            filtered_actual = []
            filtered_ids = []
            for pred, act, id_ in zip(preds_list, actual_list, ids_list):
                if id_ in valid_ids:
                    filtered_preds.append(pred)
                    filtered_actual.append(act)
                    filtered_ids.append(id_)
            
            self.preds_list.extend(filtered_preds)
            self.actual_list.extend(filtered_actual)
            self.ids.extend(filtered_ids)
        else:
            # No division filtering: extend the lists directly.
            self.preds_list.extend(preds.tolist())
            self.actual_list.extend(actual.tolist())
            ids_list = ids.tolist()
            if len(ids_list) > 1:
                self.ids.extend(ids_list)
            else:
                self.ids.append(ids.item())

    def calculate(self, preds, actual):
        self.macro_acc.update(preds, actual)
        self.micro_acc.update(preds, actual)
        self.mcc.update(preds, actual)
        self.exact_match.update(preds, actual)
        self.macro_f1.update(preds, actual)
        self.micro_f1.update(preds, actual)
        self.per_label_acc.update(preds, actual)
        self.per_label_f1.update(preds, actual)
        self.cm.update(preds, actual)

    def compute_per_label_mcc(self, preds, actual):
        """
        Compute per-label MCC by iterating over each label column using a binary MCC metric.
        """
        per_label_mcc = []
        for i in range(self.no_labels):
            binary_mcc = BinaryMatthewsCorrCoef(ignore_index=-100)
            binary_mcc.update(preds[:, i], actual[:, i])
            per_label_mcc.append(binary_mcc.compute().item())
        return per_label_mcc

    def get_metrics(self, device="cpu"):
        preds_tensor = torch.tensor(self.preds_list, device=device)
        actual_tensor = torch.tensor(self.actual_list, device=device)
        self.calculate(preds_tensor, actual_tensor)

        # Compute per-label MCC and macro MCC manually
        per_label_mcc = self.compute_per_label_mcc(preds_tensor, actual_tensor)
        macro_mcc = sum(per_label_mcc) / self.no_labels

        self.report = {
            "main": {
                "macro_accuracy": self.macro_acc.compute().item(),
                "micro_accuracy": self.micro_acc.compute().item(),
                "mcc": self.mcc.compute().item(),
                "macro_mcc": macro_mcc,
                "exact_match": self.exact_match.compute().item(),
                "macro_f1": self.macro_f1.compute().item(),
                "micro_f1": self.micro_f1.compute().item(),
                "confusion_matrix": self.cm.compute().tolist(),
                "per_label_accuracy": self.per_label_acc.compute().tolist(),
                "per_label_f1": self.per_label_f1.compute().tolist(),
                "per_label_mcc": per_label_mcc,
            },
            "pred_data": {
                "preds": self.preds_list,
                "actual": self.actual_list,
                "ids": self.ids,
            },
        }
        return self.report
