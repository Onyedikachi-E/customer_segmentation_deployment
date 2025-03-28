import joblib
import numpy as np
import pandas as pd
from model.model import AutoEncoder
from http import HTTPStatus
from sklearn.cluster import KMeans
from werkzeug.exceptions import HTTPException
from keras._tf_keras.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class SegmentationModel():
    
    def __init__(self):
        pass


    def load_label_encoder(self, features_data):
        """
        - This class method will load the saved label encoder transformer.
        - It will connvert all the categorical values to numerical values.
        """
        try:
            # Load the Label Encoder Transformer from the saved models
            onehot_encoder:OneHotEncoder = joblib.load(filename="saved_models/onehot_encoder.pkl")

            # Apply encoding on multiple categorical columns
            encoded_data = onehot_encoder.transform(features_data)

            return encoded_data
        except Exception as err:
            print(f"Error due to {str(err)}")


    def load_standard_scaler(self, encoded_data):
        """
        - This class method will load the saved Standrad Scaler Function. 
        - This is to esnure the new data is scaled the same way as the training data.
        """
        try:
            # Load the Scaler Transformer from saved models
            scaler:StandardScaler = joblib.load(filename="saved_models/scaler.pkl")

            # Standardize the data
            scaled_data = scaler.transform(encoded_data)

            return scaled_data
        except Exception as err:
            print(f"Error due to {str(err)}")
    

    def load_autoencoder_model(self, scaled_data):
        """
        - This class method will load the AutoEncoder  (ANNs) model.
        - It will be used to extract encoded features from the scaled dataset
        """
        try:
            # Load the AutoEncoder Model from the saved models
            encoder:Model = load_model("saved_models/encoder.keras")

            # Extract Encoder from Trained Autoencoder
            encoded_features = encoder.predict(scaled_data)

            return encoded_features
        except Exception as err:
            print(f"Error due to {str(err)}")
    

    def load_kmeans_model(self, encoded_features):
        """
        - This is a trained unsupervised learning model.
        - It will make prediction on the encoded features to dtermine the best fit cluster.
        """
        try:
            # Load the KMeans Model from the saved models
            kmeans:KMeans = joblib.load(filename="saved_models/kmeans.pkl")

            # Make Prediction on the encoded features to determine the cluster for the customer
            cluster = kmeans.predict(encoded_features)

            return cluster
        except Exception as err:
            print(f"Error due to {str(err)}")


    def model_predict(self, features_data):
        """
        - This is the full implementation of the Customer Segmentation model.
        - Here the new features is subjected through layers of logical methods.
        - Finally, the function returns with 
        """

        try:
            # Convert JSON to DataFrame
            features_data_dataframe = pd.DataFrame([features_data])
        
            # Perform data encoding on the new supplied features
            encoded_data = self.load_label_encoder(features_data=features_data_dataframe)

            # Standardize the endoded data
            scaled_data = self.load_standard_scaler(encoded_data=encoded_data)

            # Extract encoded features from the scaled data
            encoded_features = self.load_autoencoder_model(scaled_data=scaled_data)

            # Perform a predictiofeatures_datan on the encoded features to finally map the customer to a specific cluster.
            predicted_cluster = self.load_kmeans_model(encoded_features=encoded_features)

            # Convert NumPy integer to Python integer
            predicted_cluster = int(predicted_cluster[0])

            # Serialize the response of the model prediction data
            result = {
                "Predicted Cluster":predicted_cluster,
                "Interpretation":self.model_interpreter(cluster=predicted_cluster)
            }

            return result
        except Exception as err:
            print(f"Error due to {str(err)}")
            raise HTTPException(description=f"Model error: {str(err)}", response=HTTPStatus.BAD_REQUEST)
        

    def model_interpreter(self, cluster:int) -> str:
        
        # Error handling component of the helper function
        try:
        
            # Lets define a Dictionary of Model Interpretation
            interpretations = {
                0 : "Cluster Zero",
                1 : "Cluster One",
                2 : "Cluster Two",
                3 : "Cluster Three",
                4 : "Cluster Four",
                5 : "Cluster Five"
            }

            # Get the interpreted prediction from the dictionary
            prediction = interpretations.get(cluster)

            return prediction
        except Exception as err:
            print(f"This error was due to {str(err)}")




