#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code blends two classes together as a convex combination; a type of simple data augmentation
"""

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np


class ClassBlender(Layer):
    """Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  attenuation, batch_size, **kwargs):
        super(ClassBlender, self).__init__(**kwargs)
        self.supports_masking = True
        self.attenuation = attenuation
        self.batch_size = batch_size




    def call(self, inputs, training=None):
        def blended():
    
            inputs_permuted = tf.random_shuffle(inputs)
            angles = (180*(2*np.random.rand(self.batch_size)-1))*np.pi/180
            shifts = 4*(2*np.random.rand(self.batch_size, 2)-1) 
            inputs_permuted_translated = tf.contrib.image.translate(inputs_permuted, shifts)
            inputs_permuted_translated_rotated = tf.contrib.image.rotate(inputs_permuted_translated,angles)         
            inputs_adjusted = inputs_permuted_translated_rotated 
         
            inputs_adjusted = tf.clip_by_value(inputs_adjusted,-0.5,0.5)
            
            
            return (1.0-self.attenuation)*inputs + self.attenuation*inputs_adjusted
            
        
        return K.in_train_phase(blended, inputs, training=training)

    def get_config(self):
        config = {'attenuation': self.attenuation, 'batch_size':self.batch_size}
        base_config = super(ClassBlender, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
