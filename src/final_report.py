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

class FinalReport():
    def __init__(self, base_model, data_augm_model, tl_model):

        print(f'type={type(base_model)}')
        self.models = [base_model.title, data_augm_model.title, tl_model.title]
        self.loss = [base_model.loss, data_augm_model.loss, tl_model.title]

        self.accuracy = {
            'training': [
                base_model.history.history['accuracy'][-1], 
                data_augm_model.history.history['accuracy'][-1],
                tl_model.history.history['accuracy'][-1]
            ], 
            'validation': [
                base_model.history.history['val_accuracy'][-1], 
                data_augm_model.history.history['val_accuracy'][-1],
                tl_model.history.history['val_accuracy'][-1]
            ], 
            'testing': [
                base_model.accuracy, data_augm_model.accuracy, tl_model.accuracy
            ]
        }

    def output_report(self) -> None:
        report = pd.DataFrame({
            'Models': self.models,
            'Training Accuracy': self.accuracy['training'],
            'Validation Accuracy': self.accuracy['validation'],
            'Testing Accuracy': self.accuracy['testing'],
            'Testing Loss': self.loss,
        })

        # --- FINDINGS ----
        print(report)
