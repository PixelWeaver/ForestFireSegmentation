# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""ResNet50 model for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from resnet import resnet50

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.applications.resnet50.ResNet50',
              'keras.applications.ResNet50')

def ResNet50(*args, **kwargs):
  return resnet50.ResNet50(*args, **kwargs)


@keras_export('keras.applications.resnet50.decode_predictions')
def decode_predictions(*args, **kwargs):
  return resnet50.decode_predictions(*args, **kwargs)


@keras_export('keras.applications.resnet50.preprocess_input')
def preprocess_input(*args, **kwargs):
  return resnet50.preprocess_input(*args, **kwargs)