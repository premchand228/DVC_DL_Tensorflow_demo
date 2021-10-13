
import os
import joblib 
import logging
import tensorflow as tf

def get_VGG_16_model(input_shape ,model_path):
    model=tf.keras.applications.vgg16.VGG16(input_shape=input_shape,weights='imagnet',model_path=model_path,include_top=False)
    model.save(model_path)
    logging.info("base model is saved at path : {model_path}" )
    return model

def prepare_base_model(model,CLASSES,freeze_all,freeze_till,learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable=False
    elif (freeze_till is not None and freeze_till >0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable=False
    # add our fully connected layer
    flatten_in=tf.keras.layers.Flatten()(model.output)
    prediction=tf.keras.layers.Dense(units=2,    activation='softmax' )(flatten_in)
    full_model=tf.keras.models.Model(input=flatten_in,output=prediction)
    full_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),loss=tf.keras.losses.CategoricalCrossentropy,metrics=["accuracy"])
    logging.info("custom model is compiled and ready to be trained")
    full_model.summary()
    return full_model



