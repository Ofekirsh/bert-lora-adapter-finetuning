import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


class ExperimentLogger:
    def __init__(self, run_name: str, output_dir: str = "logs"):
        self.run_name = run_name
        self.output_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.metrics = []  # Stores a dict per epoch with keys: epoch, train_loss, val_loss, train_mcc, val_mcc
        self.csv_path = os.path.join(self.output_dir, "metrics.csv")

        self.all_val_probs = []  # For ROC and hist
        self.all_val_labels = []

    def log(self, epoch: int, metrics: dict):
        row = {"epoch": epoch, **metrics}
        self.metrics.append(row)

        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def accumulate_probs(self, probs, labels):
        self.all_val_probs.extend(probs)
        self.all_val_labels.extend(labels)

    def save_plots(self):
        if not self.metrics:
            return

        epochs = [m["epoch"] for m in self.metrics]
        keys = [k for k in self.metrics[0] if k != "epoch"]

        for key in keys:
            values = [m[key] for m in self.metrics]
            plt.figure()
            plt.plot(epochs, values, marker="o")
            plt.title(f"{key} over epochs")
            plt.xlabel("Epoch")
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"{key}.png"))
            plt.close()

    def save_roc_curve(self):
        if not self.all_val_probs or not self.all_val_labels:
            return

        fpr, tpr, _ = roc_curve(self.all_val_labels, self.all_val_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"))
        plt.close()

    def save_confusion_matrix(self):
        if not self.all_val_probs or not self.all_val_labels:
            return

        predicted_labels = [1 if p >= 0.5 else 0 for p in self.all_val_probs]
        cm = confusion_matrix(self.all_val_labels, predicted_labels)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

    def save_probability_histogram(self):
        if not self.all_val_probs:
            return

        plt.figure()
        plt.hist(self.all_val_probs, bins=20, edgecolor='black')
        plt.title("Histogram of Predicted Probabilities")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "probability_histogram.png"))
        plt.close()

    def save_accuracy_plot(self):
        """Create a combined plot showing both training and validation accuracy over epochs."""
        if not self.metrics:
            return

        epochs = [m["epoch"] for m in self.metrics]

        # Extract accuracy values (handle cases where accuracy might not be present in early epochs)
        train_accuracies = []
        val_accuracies = []

        for m in self.metrics:
            train_acc = m.get("train_accuracy", None)
            val_acc = m.get("val_accuracy", None)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        # Filter out None values and corresponding epochs
        train_data = [(e, acc) for e, acc in zip(epochs, train_accuracies) if acc is not None]
        val_data = [(e, acc) for e, acc in zip(epochs, val_accuracies) if acc is not None]

        if not train_data and not val_data:
            return

        plt.figure(figsize=(10, 6))

        if train_data:
            train_epochs, train_accs = zip(*train_data)
            plt.plot(train_epochs, train_accs, marker="o", label="Training Accuracy", color="blue")

        if val_data:
            val_epochs, val_accs = zip(*val_data)
            plt.plot(val_epochs, val_accs, marker="s", label="Validation Accuracy", color="red")

        plt.title("Training and Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)  # Accuracy is between 0 and 1
        plt.savefig(os.path.join(self.output_dir, "accuracy_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def finalize(self):
        self.save_plots()
        self.save_roc_curve()
        self.save_accuracy_plot()
        self.save_confusion_matrix()
        self.save_probability_histogram()
