import os
import albumentations as A
import tensorflow as tf

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    _bytes = tf.train.BytesList(value=[value])
    return tf.train.Feature(bytes_list= _bytes )

def _float_feature(value):
    _float = tf.train.FloatList(value=[value])
    return tf.train.Feature(float_list= _float)

def _int64_feature(value):
    _int64 = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list= _int64)


class Make_generator():
    def __init__(self  , tfr_dir  , batch_size , shuffle_num , transform_train , transfrom_val ,  fold = 5 ):
        data = os.listdir(tfr_dir)
        data = [os.path.join(tfr_dir , data_i ) for data_i in data]
        self.tfr_data = data
        self.batch_size = batch_size
        self.shuffle_num = shuffle_num
        self.transform_train = transform_train
        self.transform_val = transfrom_val
        self.fold = fold

    def __doc__(self):
        '''
        make_tfrecord방식으로 만든 tfrecord를 읽어오고 generator를 생성하는class
        '''

    #image를 받아서 aug_img로 return해주는 함수
    def aug_fn(self , image ):
        data = {"image": image}
        aug_data = self.transform_train(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img / 255.0, tf.float32)
        return aug_img

    def aug_fn_test(self , image ):
        data = {"image": image}
        aug_data = self.transform_val(**data)
        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img / 255.0, tf.float32)
        return aug_img

    # tfrecord file을 data로 parsing해주는 function
    def _parse_function(self , tfrecord_serialized):
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'cls_num': tf.io.FixedLenFeature([], tf.int64)
                    }
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [448, 448, 3])
        aug_img = tf.numpy_function(func=self.aug_fn, inp=[image], Tout=tf.float32)

        label = tf.cast(parsed_features['cls_num'], tf.int64)

        return aug_img, label

    def _parse_function_test(self , tfrecord_serialized):
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'cls_num': tf.io.FixedLenFeature([], tf.int64)
                    }

        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [448 ,448 , 3])
        aug_img = tf.numpy_function(func=self.aug_fn_test, inp=[image], Tout=tf.float32)

        label = tf.cast(parsed_features['cls_num'], tf.int64)

        return aug_img, label

    #dataset을 loding
    def load(self ):
        tfr_data = self.tfr_data
        tfr_val = tfr_data[self.fold-1]
        del tfr_data[self.fold-1]
        tfr_train = tfr_data
        train_dataset = tf.data.TFRecordDataset(tfr_train)
        train_dataset = train_dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.prefetch(
            tf.data.experimental.AUTOTUNE).batch(self.batch_size).shuffle(self.shuffle_num)

        ## validation dataset 만들기
        val_dataset = tf.data.TFRecordDataset(tfr_val)
        val_dataset = val_dataset.map(self._parse_function_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size)
        return train_dataset , val_dataset


# if __name__ =='__main__':
#     transforms_train = A.Compose([
#         A.Resize(224, 224)
#     ])
#
#     transforms_val = A.Compose([A.Resize(224, 224)
#                                 ])
#     make_generator = Make_generator(tfr_dir = '../dataset/tfrecord' , batch_size =  64 , shuffle_num = 5000 ,
#                                         transform_train  = transforms_train, transfrom_val = transforms_val ,  fold = 4)
#     train_generator , val_generator = make_generator.load( )
#
#     print('end')
#     data_val = next(iter(val_generator))
#     for a ,b in train_generator:
#         print(b)

