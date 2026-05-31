# final_report.py

"""
==================================
 FINAL RESULT METRICS
==================================

Training Accuracy: >= 90%
Validation Accuracy: >= 85%
Testing Accuracy: >= 85%
Testing Loss: <= 50% (want it closes to 0)
"""

import pandas as pd
from tensorflow.keras.models import Model
from src.utils import show_banner


class FinalReport():
    def __init__(self, models) -> None:

        self.model_titles = [model.title for model in models]
        self.loss = [model.loss for model in models]

        self.accuracy = {
            'training': [self.__get_metric(model, 'accuracy') for model in models],
            'validation': [self.__get_metric(model, 'val_accuracy') for model in models],
            'testing': [model.accuracy for model in models]
        }

    def __get_metric(self, model_obj, metric_name: str) -> float:
        # Safely retrieves the last value of a metric from the history object.
        if model_obj.history and hasattr(model_obj.history, 'history'):
            if metric_name in model_obj.history.history:
                return model_obj.history.history[metric_name][-1]
        return 0.0

    def pick_winner(self, report) -> str:
        """
        Testing Accuracy and Testing Loss are the most important metrics
        To determine the true winner, you must look at the Per-Class Precision, Recall, and F1-Scores generated in your
        evaluation reports:
        F1-Score: The harmonic mean of Precision and Recall. The model with the highest macro-averaged F1-Score across
        all 12 classes is your true winner.
        """

        return ''


    def calc_model_score(self, y_true_enc, y_pred_probs, alpha=1.0, beta=1.0):
        # Convert one-hot vectors to 1D integer class arrays
        y_true = np.argmax(y_true_enc, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 1. Compute Macro F1-Score (unweighted mean across all classes)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # 2. Get the full confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate global False Positives and False Negatives across all classes
        # FP: Column sums minus diagonal (predicted as class X, but wasn't)
        # FN: Row sums minus diagonal (was class X, but predicted as something else)
        fp_count = np.sum(cm, axis=0) - np.diag(cm)
        fn_count = np.sum(cm, axis=1) - np.diag(cm)

        total_fp = np.sum(fp_count)
        total_fn = np.sum(fn_count)

        # 3. Apply the Weighted Penalty Score formula
        performance_score = (100 * macro_f1) - ((alpha * total_fp) + (beta * total_fn))

        print(f"--- Score Breakdown ---")
        print(f"Macro F1-Score:     {macro_f1:.4f}")
        print(f"Total False Pos:    {total_fp}")
        print(f"Total False Neg:    {total_fn}")
        print(f"FINAL MODEL SCORE:  {performance_score:.2f}")

        return performance_score

    def results(self) -> None:
        report = pd.DataFrame({
            'Models': self.model_titles,
            'Training Accuracy': self.accuracy['training'],
            'Validation Accuracy': self.accuracy['validation'],
            'Testing Accuracy': self.accuracy['testing'],
            'Testing Loss': self.loss,
        })

        # --- FINDINGS ----
        show_banner('Final Report')
        print(report)

        # Pick Winner
        winning_model = self.pick_winner(report)
        print(f'*** Winning Model is 🏆 {winning_model}. ***')


