# final_report.py

"""
FINAL RESULT METRICS
--------------------
- Training Accuracy: >= 90%
- Validation Accuracy: >= 85%
- Testing Accuracy: >= 85%
- Testing Loss: <= 50% (want it closes to 0)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.models import Model
from src.utils import show_banner


class FinalReport():
    def __init__(self, models) -> None:
        self.models = models
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

    def get_all_scores(self) -> dict:
        """
        Calculates performance scores for all models.
        """
        model_scores = {}
        for model in self.models:
            y_pred_probs = model.model.predict(model.x_test_norm)
            model_scores[model.title] = self.calc_model_score(model.title, model.y_test_enc, y_pred_probs)
        return model_scores

    def pick_best_model(self, model_scores: dict) -> str:
        """
        Identifies the winning model from a dictionary of scores.
        """
        if not model_scores:
            return 'No Models Evaluated'

        # Find the model title with the highest performance score
        return max(model_scores, key=model_scores.get)

    def calc_model_score(
        self,
        model_title: str,
        y_true_enc: np.ndarray,
        y_pred_probs: np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        # Convert one-hot vectors to 1D integer class arrays
        y_true = np.argmax(y_true_enc, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Compute Macro F1-Score (unweighted mean across all classes)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        # Get the full confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Calculate global False Positives and False Negatives across all classes
        fp_count = np.sum(cm, axis=0) - np.diag(cm)
        fn_count = np.sum(cm, axis=1) - np.diag(cm)

        total_fp = np.sum(fp_count)
        total_fn = np.sum(fn_count)

        # Find out exactly how many images were tested total
        total_samples = len(y_true)

        # Calculate the percentage of data that resulted in errors
        fp_percentage = (total_fp / total_samples) * 100
        fn_percentage = (total_fn / total_samples) * 100

        # Apply the weights to the error percentages instead of raw counts
        macro_pct = macro_f1 * 100
        penalty_costs = (alpha * fp_percentage) + (beta * fn_percentage)
        performance_score = macro_pct - penalty_costs

        print(f"\n--- 📊 {model_title} Score Breakdown 📊 ---")
        print(f"Macro F1-Score:           {macro_f1:.4f}")
        print(f"Total Test Images:        {total_samples}")
        print(f"Total False Pos (Raw):    {total_fp} ({fp_percentage:.2f}%)")
        print(f"Total False Neg (Raw):    {total_fn} ({fn_percentage:.2f}%)")
        print(f"Global Macro F1 Baseline: {macro_pct:.2f}")
        print(f"Total Penalties Applied:  -{penalty_costs:.2f}")
        print('--------------------------------')
        print(f"💡 FINAL MODEL SCORE:        {performance_score:.2f}\n")

        return performance_score

    def results(self) -> None:
        report = pd.DataFrame({
            'Models': self.model_titles,
            'Training Accuracy': self.accuracy['training'],
            'Validation Accuracy': self.accuracy['validation'],
            'Testing Accuracy': self.accuracy['testing'],
            'Testing Loss': self.loss,
        })

        # --- Final Report --- #
        show_banner('📈 FINAL REPORT 📉')
        print(report)

        # --- Evaluate and Pick Best Model --- #
        all_scores = self.get_all_scores()
        best_model_title = self.pick_best_model(all_scores)
        best_model_score = all_scores[best_model_title]

        show_banner('🏆 BEST MODEL 🏆')
        print(f'The best_model is: {best_model_title}')
        print(f'Performance Score: {best_model_score:.2f}')
