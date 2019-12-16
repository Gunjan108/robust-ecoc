#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np


class Grayscaler(Layer):
    """Converts input to grayscale
    Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  **kwargs):
        super(Grayscaler, self).__init__(**kwargs)
        self.supports_masking = True



    def call(self, inputs, training=None):
        def augmented():            
            return tf.image.rgb_to_grayscale(inputs)
                        
        return K.in_train_phase(augmented, augmented, training=training)
    
    
    

    def get_config(self):
        config = {}
        base_config = super(Grayscaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
