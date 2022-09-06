from tensorflow import keras
import numpy as np


class MLP(object):
    def __init__(self):
        # path_to_artifacts = "../../research/"
        path_to_artifacts = "./research/"
        self.model = keras.models.load_model(path_to_artifacts + "model_90_1000000_all")
        self.model.summary()

    @staticmethod
    def preprocessing(input_data):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        return input_data

    def predict(self, input_data):
        y_pred_list = self.model.predict(input_data)
        label = np.array(y_pred_list).argmax(axis=1).tolist()
        probability = [y_pred_list[i][label[i]]for i in range(len(label))]

        return {"probability": probability, "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            preprocessed_data = self.preprocessing(input_data)
            result_dict = self.predict(preprocessed_data)  # only one sample
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return result_dict

