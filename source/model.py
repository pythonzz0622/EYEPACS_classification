from tensorflow.keras.layers import Input , GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf


class InceptionV3(tf.keras.Model):
    def __init__(self , classifier ):
        super(InceptionV3, self).__init__()
        self.model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(299, 299, 3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        self.pool = GlobalAveragePooling2D()
        ##ouput 2048
        self.classifier = classifier

        # self.classifier = keras.Sequential([
        #
        #     Dropout(0.5),
        #     Dense(2, activation='softmax', name='output')
        # ])

    def call(self, inputs):
        x = self.model(inputs)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def summary(self, input_shape):
        inputs = Input(input_shape)
        Model(inputs, self.call(inputs)).summary()


