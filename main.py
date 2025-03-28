from flask import abort
from flask import Flask
from flask import request
from flask import jsonify
from http import HTTPStatus
from flask import render_template
from model.model import AutoEncoder
from model.model import CustomerData
from werkzeug.exceptions import HTTPException
from repository.customer_segmentation import SegmentationModel
import warnings

warnings.simplefilter("ignore", category=UserWarning)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def model_predict():
    """
    This route handles the request of features of customers
    """
    try:
        # Extract features from the request JSCollection body
        features_data = request.get_json()

        print(features_data)
        
        if not features_data:
            return jsonify({"message": "Customer Feature information was not supplied"}), HTTPStatus.BAD_REQUEST
        
        # Initiate the Customer Segmentation instance
        segmentation_repo = SegmentationModel()
        
        # Make Prediction with the provided data using the trained Model.
        prediction = segmentation_repo.model_predict(features_data=features_data)
        # prediction = "We are good"
        return jsonify(prediction), HTTPStatus.CREATED
    
    except Exception as err:
        raise HTTPException(description=f"Database error: {str(err)}", response=HTTPStatus.BAD_REQUEST)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, debug=True)