from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score,
                             precision_recall_curve, accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)
import pandas as pd
import numpy as np
from ultralytics import YOLO

class ModelAnalyzer:
    def __init__(self, model_path, results_csv_path, labels):
        self.model = YOLO(model_path)
        self.results_csv_path = results_csv_path
        self.labels = labels
    
    def plot_loss_curves(self):
        df = pd.read_csv(self.results_csv_path)
        df.columns = df.columns.str.strip()

        plt.figure(figsize=(10, 5))
        plt.plot(df['train/box_loss'], label='Box Loss')
        plt.plot(df['train/cls_loss'], label='Class Loss')
        plt.plot(df['train/dfl_loss'], label='DFL Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.labels)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def get_predictions(self, test_data_path):
        predictions = self.model.predict(source=test_data_path, save=False)
        y_true, y_pred, y_prob = [], [], []

        for pred in predictions:
            y_true.extend(pred.boxes.cls.numpy())
            y_pred.extend(pred.boxes.cls.numpy())
            y_prob.extend(pred.boxes.conf.numpy())

        self.plot_confusion_matrix(y_true, y_pred)
        return y_true, y_pred, y_prob

    def plot_roc_curve(self, y_true, y_prob, pos_label=1):
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
        print("AUC:", roc_auc)

    def plot_precision_recall_curve(self, y_true, y_prob, pos_label=1):
        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    def calculate_map(self):
        results = self.model.val()
        print("mAP@50:", results.maps[0])  
        print("mAP@50-95:", results.maps)

    def calculate_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        print("F1 Score (weighted):", f1)

    def calculate_accuracy(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

    def calculate_balanced_accuracy(self, y_true, y_pred):
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        print("Balanced Accuracy:", balanced_acc)

    def calculate_cohens_kappa(self, y_true, y_pred):
        kappa = cohen_kappa_score(y_true, y_pred)
        print("Cohen's Kappa:", kappa)

    def calculate_mcc(self, y_true, y_pred):
        mcc = matthews_corrcoef(y_true, y_pred)
        print("Matthews Correlation Coefficient (MCC):", mcc)

    def calculate_dice_coefficient(self, y_true, y_pred, average='weighted'):
        dice_coefficient = f1_score(y_true, y_pred, average=average)
        print("Dice Coefficient:", dice_coefficient)

    def analyze_model_performance(self, test_data_path, pos_label=1):
        print("1. Kayıp (Loss) Eğrileri")
        self.plot_loss_curves()

        y_true, y_pred, y_prob = self.get_predictions(test_data_path)

        print("2. ROC Eğrisi ve AUC")
        self.plot_roc_curve(y_true, y_prob, pos_label)

        print("3. Precision-Recall Eğrisi")
        self.plot_precision_recall_curve(y_true, y_prob, pos_label)

        print("4. mAP Hesaplama")
        self.calculate_map()

        print("5. F1 Skoru Hesaplama")
        self.calculate_f1_score(y_true, y_pred)

        print("6. Doğruluk (Accuracy) Hesaplama")
        self.calculate_accuracy(y_true, y_pred)

        print("7. Dengelenmiş Doğruluk (Balanced Accuracy) Hesaplama")
        self.calculate_balanced_accuracy(y_true, y_pred)

        print("8. Cohen's Kappa Skoru Hesaplama")
        self.calculate_cohens_kappa(y_true, y_pred)

        print("9. Matthews Correlation Coefficient (MCC) Hesaplama")
        self.calculate_mcc(y_true, y_pred)

        print("10. Dice Coefficient Hesaplama")
        self.calculate_dice_coefficient(y_true, y_pred)

model_path = "/Users/bilge/Desktop/runs/detect/Enfermedades_cerebro_yolo85/weights/best.pt"
results_csv_path = "/Users/bilge/Desktop/runs/detect/Enfermedades_cerebro_yolo85/results.csv"
test_data_path = "/Users/bilge/Downloads/yolov11/test/images"
labels = ['Alzheimer', 'CONTROL', 'parkinson']

analyzer = ModelAnalyzer(model_path, results_csv_path, labels)
analyzer.analyze_model_performance(test_data_path)