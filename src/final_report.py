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


