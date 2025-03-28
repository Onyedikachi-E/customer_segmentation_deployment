


from customer_segmentation import SegmentationModel
import joblib
import pandas as pd


customer_data = {
    "Product_Name": ["PulseDesk"],
    "Products_Category": ["SaaS"],
    "Subscription_Type": ["Pro"],
    "Usage_Frequency": ["Weekly"],
    "Payment_Plan": ["Annual"],
    "Marketing_Channel": ["Social Media"]
}



def load_label_encoder(features_data):
        """
        - This class method will load the saved label encoder transformer.
        - It will connvert all the categorical values to numerical values.
        """

        print(features_data)

        # Load the Label Encoder Transformer from the saved models
        onehot_encoder = joblib.load(filename="saved_models/onehot_encoder.pkl")

        print("label_encoder")

        features_data = onehot_encoder.transform(features_data[["Product_Name", "Products_Category", "Subscription_Type", "Usage_Frequency", "Payment_Plan", "Marketing_Channel"]])

        # Perform data encoding on the featured data to numerical values
        # encoded_data = onehot_encoder.fit_transform(features_data)

        print(features_data)

        return "encoded_data"


def load_standard_scaler(encoded_data):
        """
        - This class method will load the saved Standrad Scaler Function. 
        - This is to esnure the new data is scaled the same way as the training data.
        """
        try:
            # Load the Scaler Transformer from saved models
            scaler = joblib.load(filename="saved_models/scaler.pkl")

            print(encoded_data)

            # Standardize the data
            scaled_data = scaler.fit_transform(encoded_data)

            print(scaled_data)

            return "scaled_data"
        except Exception as err:
            print(f"Error due to {str(err)}")
    


# segmentation_repo = SegmentationModel()

customer_data = pd.DataFrame(customer_data)
encoded_data = load_label_encoder(features_data=customer_data)
load_standard_scaler(encoded_data)