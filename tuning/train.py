import sys 
sys.path.append('/home/user304/users/jiwon/PAI_EYEPACS/source')
import tqdm
import numpy as np
import pandas as pd
from sympy import arg
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import albumentations as A
import os
import cv2
import warnings
warnings.filterwarnings(action='ignore')
import mlflow
import Loader
import model
import utils
import argparse
import json
import yaml


parser = argparse.ArgumentParser(description='get config')
parser.add_argument('--layers', type=yaml.load , help='json dir')
args = parser.parse_args()
classifier_args = args.layers
mlflow.set_tracking_uri('./mlruns')
mlflow.set_experiment(experiment_name= 'set_classifier_v2')


#Augumentation부분
transforms_train = A.Compose([
                            A.Resize(299, 299) , 
                            A.Normalize( mean = (  0.485, 0.456, 0.406) , 
                                         std = (0.229, 0.224, 0.225) ) 
])
transforms_val = A.Compose([
                            A.Resize(299, 299) ,
                            A.Normalize( mean = (  0.485, 0.456, 0.406) , 
                                         std = (0.229, 0.224, 0.225) )
                            ])

### DataGenerator부분
make_generator = Loader.Make_generator(tfr_dir = '../dataset/tfrecord' , batch_size =  128 , shuffle_num = 5000 ,
                                    transform_train  = transforms_train, transfrom_val = transforms_val ,  fold = 5)

train_generator , val_generator = make_generator.load()


## model 정의 부분
'''
layers_list = []
for k , v in  classifier_args.items():
    if k.split('_')[1] == 'hidden':
        layer = layers.Dense(v)
    if k.split('_')[1] == 'dropout':
        layer = layers.Dropout(v)
layers_list.append(layer)
classifier = keras.Sequential(layers_list)
'''
classifier = keras.Sequential([
                                            layers.Dropout(0.5) , 
                                            layers.Dense(2 , activation= 'softmax' , name = 'output')
                                            ])

model = model.InceptionV3(classifier = classifier)

### optimizer 정의 부분
optimizer = utils.set_optimizer('SGD' , learning_rate = 0.1, momentum = 0.3)
#### 학습부분
loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        pred = model(images, training=True)
        loss = loss_function(labels, pred)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    correct = (tf.math.argmax(pred , axis = 1 ) == labels)
    return loss , correct


@tf.function
def val_step(images, labels):

    pred = model(images, training=False)
    loss = loss_function(labels, pred)
    correct = (tf.math.argmax(pred , axis = 1 ) == labels)
    return loss , correct


best_acc = 0.0
best_loss = 0.0
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

mlflow.end_run()
with mlflow.start_run(run_name = 'remove_all_aug') as run:

    for epoch in tqdm.tqdm(range(1 ,200)):

        size = 0
        train_loss = 0
        train_acc = 0

        for i, (images, labels) in enumerate(train_generator):
            loss , correct = train_step(images, labels)

            size += labels.shape[0]
            train_loss += loss.numpy().sum()
            train_acc += correct.numpy().sum()

        train_loss /= (i+1)
        train_acc /= size

        size = 0
        val_loss = 0
        val_acc = 0
        for j ,(images, labels) in enumerate(val_generator):
            loss , correct = val_step(images, labels)

            size += labels.shape[0]
            val_loss += loss.numpy().sum()
            val_acc += correct.numpy().sum()

        val_loss /= (j + 1)
        val_acc /= size
        if epoch % 50 == 0 :
            template = 'epoch: {} , loss: {:.3f} , acc: {:.3f} , val_loss: {:.3f} , val_acc : {:.3f}'
            print(template.format(epoch + 1, train_loss, train_acc * 100, val_loss, val_acc * 100))

        if val_acc > best_acc:
            best_train_acc = train_acc
            best_train_loss = train_loss
            best_acc = val_acc
            best_loss = val_loss
            best_metric = {'train_acc' : train_acc  , 'val_acc' : val_acc , 'train_loss' : train_loss , 'val_loss' : val_loss}
            best_params = { 'optimizer' : optimizer, 
                            'augmodel' : transforms_train , 
                            'classifier' : classifier
                            }
            best_model = model
            # best_model.save_weights(os.path.join('../models', 'DRD_{}.h5'.format(epoch)))

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
    print('end')
    metrics = {'train_loss' : train_loss_list , 'train_acc' : train_acc_list, 
            'val_acc' : val_acc_list , 'val_loss' : val_loss_list  }

    utils.save_plot(metrics , 'epoch200' , dir = './artifact/')


    mlflow.set_tag('epoch', '200' )
    mlflow.set_tag('add' , 'augmentations')
    mlflow.log_artifacts('./artifact/')
    mlflow.log_metrics(best_metric)

    # mlflow.log_params( classifier_args)
#     mlflow.tensorflow.save_model('../')
#     mlflow.tensorflow.save_model(tf_saved_model_dir = '../models/' , tf_meta_graph_tags = )