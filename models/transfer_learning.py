# transfer_learning.py
from models.cnn_model import CnnModel

"""
Transfer Learning Model

The purpose of using a Transfer Learning Model (TLM) is to leverage knowledge gained from a massive, general task to 
solve a specific, smaller task.
"""
class TransferLayerModel(CnnModel):
    def __init__(self, params):
        self.title = 'Transfer Learning Model'
        self.image_params = params
