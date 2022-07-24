import tqdm
import tensorflow as tf
import warnings
warnings.filterwarnings(action='ignore')
import utils
from sklearn.metrics import roc_auc_score
tf.config.run_functions_eagerly(True)
class training():
    '''
    training 시켜주는 모듈
    '''
    def __init__(self , model , optimizer ,loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def _train_step(self , images, labels):
        with tf.GradientTape() as tape:
            pred = self.model(images, training=True)
            loss = self.loss_fn(labels, pred)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        correct = (tf.math.argmax(pred , axis = 1 ) == labels)
        auc = roc_auc_score(labels, pred.numpy()[:, 1] )
        return loss , correct , auc 

    @tf.function
    def _val_step(self , images, labels):

        pred = self.model(images, training=False)
        loss = self.loss_fn(labels, pred)
        correct = (tf.math.argmax(pred , axis = 1 ) == labels)
        auc = roc_auc_score(labels, pred.numpy()[:, 1] )
        return loss , correct , auc

    def training(self , train_generator ,val_generator  , epoch ):
        best_val_acc = 0.0
        best_train_acc = 0.0
        best_val_auc = 0.0
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        

        for epoch in tqdm.tqdm(range(1 ,epoch)):

            size = 0
            train_loss = 0
            train_acc = 0
            train_auc = 0.0

            for i, (images, labels) in enumerate(train_generator):
                loss , correct , auc  = self._train_step(images, labels)

                size += labels.shape[0]
                train_loss += loss.numpy().sum()
                train_acc += correct.numpy().sum()
                train_auc += auc

            train_loss /= (i+1)
            train_acc /= size
            train_auc /= (i + 1)

            size = 0
            val_loss = 0
            val_acc = 0
            val_auc = 0.0
            for j ,(images, labels) in enumerate(val_generator):
                loss , correct , auc = self._val_step(images, labels)

                size += labels.shape[0]
                val_loss += loss.numpy().sum()
                val_acc += correct.numpy().sum()
                val_auc += auc

            val_loss /= (j + 1)
            val_auc /= (j + 1)
            val_acc /= size

            if epoch % 50 == 0 :
                template = 'epoch: {} , loss: {:.3f} , acc: {:.3f} , val_loss: {:.3f} , val_acc : {:.3f}'
                print(template.format(epoch + 1, train_loss, train_acc * 100, val_loss, val_acc * 100))

            if val_auc > best_val_auc:
                best_train_acc = train_acc
                best_train_loss = train_loss
                best_train_auc = train_auc
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_val_loss = val_loss

                best_metrics = {'train_acc' : best_train_acc  , 'val_acc' : best_val_acc , 'train_loss' : best_train_loss , 'val_loss' : best_val_loss , 
                                'train_auc' : best_train_auc  , 'val_auc' : best_val_auc}
                best_model = self.model
                # best_model.save_weights(os.path.join('../models', 'DRD_{}.h5'.format(epoch)))
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            
        print('end')
        metrics = {'train_loss' : train_loss_list , 'train_acc' : train_acc_list,  
                'val_acc' : val_acc_list , 'val_loss' : val_loss_list   }
        print(best_metrics)
        print(best_model)
        utils.save_plot(metrics , 'epoch200' , dir = './artifact/')

        return best_model , best_metrics