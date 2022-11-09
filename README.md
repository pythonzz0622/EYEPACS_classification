# EYEPACS_practice

##### 결과확인 [MLFLOW]



[MLFLOW]: http://203.255.39.106:5000/#/experiments/2/

[Paper review] : https://flaxen-lead-f97.notion.site/Paper-Review-d97220a583e1477b86d33937639673fe

<img width="404" alt="image" src="https://user-images.githubusercontent.com/90737305/200765943-ca5fd197-053f-4cba-9d47-dc04647bd2c7.png">

<img width="249" alt="image" src="https://user-images.githubusercontent.com/90737305/200771066-ffe090d7-9df5-4176-9c47-bd8219831c0a.png">



망막 병증 부분을 더 명확히 구분 하기 위해서 CLAHE 함수 적용


<img width="183" alt="image" src="https://user-images.githubusercontent.com/90737305/200769417-a08af6b7-e404-4405-9868-b281a7baab76.png">
<img width="183" alt="image" src="https://user-images.githubusercontent.com/90737305/200767980-6308f19a-aef0-4636-9b90-6558656d0f5b.png">

<img width="332" alt="image" src="https://user-images.githubusercontent.com/90737305/200767763-2954a1bd-2368-45e3-8879-dd129373786f.png">


source
make_tfrecord -> data format을 tfrecord로 변환함, 독립적으로 실행 가능

make_model.py -> inceptionV3 model 생성하는 package

Loader.py -> tfrecord file을 decoding하고 generator를 생성하는 source

train_module -> Gradient Tape을 활용해 loss를 계산하고 optimizing하는 source

utils.py -> optimizer setting과 plotting등 여러가지 util source가 들어가 있음


train_module.py -> make_model , Loader , train_module, utils를 활용하여 실제로 학습되는 부분

bash file을 활용해 train_v2.py 파일 자동화



