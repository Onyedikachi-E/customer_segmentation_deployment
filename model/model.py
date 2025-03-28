from pydantic import BaseModel
## Load the neccesary algorithms
import tensorflow as tf
from keras._tf_keras.keras.saving import register_keras_serializable
from keras._tf_keras.keras.models import Sequential, Model, load_model
from keras import layers, optimizers, losses, metrics, utils

class CustomerData(BaseModel):
    Product_Name: str
    products_category:str
    subscription_type:str
    Usage_Frequency: str
    payment_plan:str
    marketing_channel:str


#######################################################################################

############ Building the Autoencoder Model Class with essential methods ##############

#######################################################################################


# Encoder Model
def build_encoder(input_dim, encoding_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(units=64, activation='relu')(inputs)
    encoded = layers.Dense(units=encoding_dim, activation='relu', name="encoded_layer")(x)
    encoded = layers.Dropout(0.2)(encoded)  # Regularization
    return Model(inputs, encoded, name="Encoder")

# Decoder Model
def build_decoder(encoding_dim, output_dim):
    inputs = layers.Input(shape=(encoding_dim,))
    x = layers.Dense(units=64, activation='relu')(inputs)
    decoded = layers.Dense(units=output_dim, name="decoded_layer")(x)
    return Model(inputs, decoded, name="Decoder")

# AutoEncoder Model
@register_keras_serializable()
class AutoEncoder(Model):
    def __init__(self, input_dim=None, encoding_dim=None, **kwargs):
        """Allow additional arguments like `trainable` to avoid deserialization errors."""
        super(AutoEncoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        if input_dim is not None and encoding_dim is not None:
            self.encoder = build_encoder(input_dim, encoding_dim)
            self.decoder = build_decoder(encoding_dim, input_dim)
        else:
            self.encoder = None
            self.decoder = None

    def call(self, X):
        """ Forward pass """
        Z = self.encoder(X)
        return self.decoder(Z)

    def encode(self, X):
        """ Encode input data """
        return self.encoder(X)

    def decode(self, Z):
        """ Decode encoded representation """
        return self.decoder(Z)

    def get_config(self):
        """ Ensure serialization support """
        config = super(AutoEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "encoding_dim": self.encoding_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """ Restore model from config """
        input_dim = config.get("input_dim", None)
        encoding_dim = config.get("encoding_dim", None)
        return cls(input_dim=input_dim, encoding_dim=encoding_dim, **config) 