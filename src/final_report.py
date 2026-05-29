# final_report.py

"""
# ==================================
#  FINAL RESULT METRICS
# ==================================

Training Accuracy: >= 90%
Validation Accuracy: >= 85%
Testing Accuracy: >= 85%
Testing Loss: <= 50% (want it closes to 0)
"""

import pandas as pd
from tensorflow.keras.models import Model
from src.utils import show_banner

"""
class FinalReport():
    def __init__(self, base_model, data_augm_model, tl_model) -> None:

        self.model_titles = [base_model.title, data_augm_model.title, tl_model.title]
        
        # Fixed typo: was using tl_model.title instead of tl_model.loss
        self.loss = [base_model.loss, data_augm_model.loss, tl_model.loss]

        self.accuracy = {
            'training': [
                self._get_metric(base_model, 'accuracy'),
                self._get_metric(data_augm_model, 'accuracy'),
                self._get_metric(tl_model, 'accuracy')
            ], 
            'validation': [
                self._get_metric(base_model, 'val_accuracy'),
                self._get_metric(data_augm_model, 'val_accuracy'),
                self._get_metric(tl_model, 'val_accuracy')
            ], 
            'testing': [
                base_model.accuracy, data_augm_model.accuracy, tl_model.accuracy
            ]
        }

    def _get_metric(self, model_obj, metric_name: str) -> float:
        # Safely retrieves the last value of a metric from the history object.
        if model_obj.history and hasattr(model_obj.history, 'history'):
            if metric_name in model_obj.history.history:
                return model_obj.history.history[metric_name][-1]
        return 0.0

    def output_report(self) -> None:
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
"""

class FinalReport():
    def __init__(self, models) -> None:

        self.model_titles = [model.title for model in models]
        self.loss = [model.loss for model in models]

        self.accuracy = {
            'training': [self._get_metric(model, 'accuracy') for model in models],
            'validation': [self._get_metric(model, 'val_accuracy') for model in models],
            'testing': [model.accuracy for model in models]
        }

    def _get_metric(self, model_obj, metric_name: str) -> float:
        # Safely retrieves the last value of a metric from the history object.
        if model_obj.history and hasattr(model_obj.history, 'history'):
            if metric_name in model_obj.history.history:
                return model_obj.history.history[metric_name][-1]
        return 0.0

    def output_report(self) -> None:
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
