import os
import pandas as pd
import tqdm
import tensorflow as tf
import cv2
import argparse

# get arguments
parser = argparse.ArgumentParser(description='get config')
parser.add_argument('--image_size', type=int, help='output image size')
parser.add_argument('--data_dir', type=str, help='data_folder dir')
parser.add_argument('--image_dir', type=str, help='image_dir  in data_folder dir')
parser.add_argument('--df_path', type=str, help='df_file_path')

args = parser.parse_args()
image_size = args.image_size
data_dir = args.data_dir
image_dir = args.image_dir
df_path = args.df_path


# data to TFFeature
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    _bytes = tf.train.BytesList(value=[value])
    return tf.train.Feature(bytes_list= _bytes )

def _float_feature(value):
    _float = tf.train.FloatList(value=[value])
    return tf.train.Feature(float_list= _float)

def _int64_feature(value):
    _int64 = tf.train.Int64List(value=[value])
    return tf.train.Feature(int64_list= _int64)

#image2btye
def _to_byte_img( image_path , image_size):
    _image = cv2.imread(image_path)
    _image = cv2.cvtColor(_image , cv2.COLOR_BGR2RGB)
    _image = cv2.resize(_image , (image_size , image_size))
    bimage = _image.tobytes()
    return bimage

#data2example
def _serialize_example(bimage , label):
    example = tf.train.Example(features = tf.train.Features(feature = {
        'image' : _bytes_feature(bimage) ,
        'cls_num' : _int64_feature(label)
    }))
    return example.SerializeToString()


class Make_tfrecord():
    def __init__(self , image_size , data_dir , image_dir , df_path ):
        
        self.image_size = image_size
        self.image_dir = image_dir 
        self.data_dir = data_dir
        self.df = df = pd.read_csv( os.path.join(data_dir  , df_path) )
        self.tf_dir  = os.path.join(data_dir  , 'tfrecord/')

    def __doc__(self):
        '''
        image path , df path를 통해
        Image와 label 데이터를 받아서 5fold fTFrecord형식으로 변환하는 class
        아래와 같은 예시로 실행하면 됨
        python make_tfrecord.py --image_size 448  --data_dir '../dataset'  --image_dir 'IMAGES'  --df_path 'data_label.csv'
        '''

    # fold 별로 추출해 내기
    def write_fold(self):

    #tf_Record 5 fold를 만드는 함수
        os.makedirs(self.tf_dir  , exist_ok= True)
        for i in range(1, 6):
            _fold_dir = os.path.join(self.tf_dir , f'fold{i}.tfr') 
            globals()[f'tf_fold_{i}_data'] = tf.io.TFRecordWriter(_fold_dir)
        print('5개 fold writer 생성완료')

        # fold 별로 tf_record 저장하는 iterator
        for i in tqdm.tqdm(range(1 , 6)):
            _df = self.df[self.df['fold'] == i]

            with globals()[f'tf_fold_{i}_data'] as writer:
                for _ , line in _df.iterrows():

                    # image2byte load2label
                    image_path = os.path.join(self.data_dir  , 'IMAGES/' , line['image_id'])
                    bimage = _to_byte_img(image_path , self.image_size)
                    class_num = line['target']

                    # byteimg , label to example
                    example = _serialize_example(bimage , class_num )
                    writer.write(example)
        
        print('tfr생성완료' ,os.listdir(self.tf_dir))

if __name__=='__main__':
    make_tfrecord = Make_tfrecord(image_size , data_dir , image_dir ,df_path)
    make_tfrecord.write_fold()
