import sys 
sys.path.append('../source')
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import albumentations as A
import warnings
warnings.filterwarnings(action='ignore')
import mlflow

import Loader , make_model , utils  , train_module
import argparse , yaml

### format python train_v2.py --run_name "optimization" --addtag "fist:experiment"
parser = argparse.ArgumentParser(description='get config')
parser.add_argument('--run_name' , type = str , help =  'add_run_name')
parser.add_argument('--layers', type=yaml.load , help='json dir')
parser.add_argument('--addtag' , type = str ,  help = 'alpha:beta')
parser.add_argument('--optimizer' , type = str , help = 'set optimizer')
parser.add_argument('--lr' , type = float , help ='lr')
args = parser.parse_args()

##########################     set hyperparmeters  ###############################

RUN_NAME = args.run_name
classifier_args = args.layers
_addtag = args.addtag
TAG = _addtag.split(':')
EPOCH = 200
LR = args.lr
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LR,
    decay_steps=10000,
    decay_rate=0.9)

OPTIMIZER = utils.set_optimizer(args.optimizer , learning_rate = lr_schedule )
LOSS_FN = tf.keras.losses.SparseCategoricalCrossentropy() 

##########################     make data augmentation    ###############################

transforms_train = A.Compose([
                            A.Resize(299, 299) ,
                            A.CLAHE(always_apply= True),
                            A.RandomBrightness(),
                            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
                            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                            A.HorizontalFlip(),
                            A.Normalize( mean = (  0.485, 0.456, 0.406) , 
                                         std = (0.229, 0.224, 0.225) ) 
])
transforms_val = A.Compose([
                            A.Resize(299, 299) ,
                            A.CLAHE(always_apply = True),
                            A.Normalize( mean = (  0.485, 0.456, 0.406) , 
                                         std = (0.229, 0.224, 0.225) )
                            ])

##########################     make data generator  ###############################

make_generator = Loader.Make_generator(tfr_dir = '../dataset/tfrecord' , batch_size =  128 , shuffle_num = 5000 ,
                                    transform_train  = transforms_train, transfrom_val = transforms_val ,  fold = 5)
train_generator , val_generator = make_generator.load()


##########################     model setting part   ###############################

classifier = keras.Sequential([
                                            layers.Dropout(0.5) , 
                                            layers.Dense(2 , activation= 'softmax' , name = 'output')
                                            ])
model = make_model.InceptionV3(classifier = classifier)


##########################      model training part   ###############################

trainer  = train_module.training(model , OPTIMIZER , LOSS_FN)


mlflow.set_tracking_uri('./mlruns')
mlflow.set_experiment(experiment_name= 'testinput')
mlflow.end_run()
with mlflow.start_run(run_name = f'{args.optimizer}_{LR}') as run:
    ##run train_module4
    
    best_model , best_metrics = trainer.training(train_generator , val_generator , EPOCH)
    mlflow.set_tag(TAG[0], TAG[1] )
    mlflow.log_artifacts('./artifact/')
    mlflow.log_metrics(best_metrics)

    # mlflow.log_params( classifier_args)
#     mlflow.tensorflow.save_model('../')
#     mlflow.tensorflow.save_model(tf_saved_model_dir = '../models/' , tf_meta_graph_tags = )