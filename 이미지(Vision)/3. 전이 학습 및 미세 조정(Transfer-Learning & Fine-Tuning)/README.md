# 코드 설명

## 챕터 설명
### 전이학습 및 미세 조정이란
#### 전이학습
- **전이학습**이란 **이미 학습된 모델**을 가져다 **새로운 문제에 활용**하는 것을 말한다.

#### 미세조정
- 가져온 모델의 **일부 층**을 **조금 더 학습시켜서 조정**하는 것을 말한다.

### 이 챕터에서는..
- 이 챕터에서는 **이미 학습된 모델(``MobileNetV2``)** 을 가져와 **미세조정**을 하여 ``개``와 ``고양이``를 구분하는 모델을 생성한다.


## 데이터셋 다운로드
```python
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
```
- 경로를 통해서 개와 고양이 데이터 이미지를 다운로드 한다.

## 데이터셋 생성
### 훈련 데이터셋
```python
# 훈련 데이터셋
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)
```

### 검증 데이터셋
```python
# 검증 데이터셋
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)
```

### 테스트 데이터셋 
```python
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
```
- **검증 데이터셋**의 ``20%`` 만큼 들고와서 **테스트 데이터셋**을 생성한다.





<!--
특징 추출 = {
    컨벌루션 베이스 모델을 동결하고 특징 추출기로 사용한다. 는 
    학습을 안 시키고 특징 추출기로만 사용한다는 뜻이다.
    
    이유 {
        이미 수천 수만장 학습이 끝난 모델이기 때문에 특징만 추출하면 되기 때문
    },

    incldue_top=False {
        모델을 다운할 때 분류층을 포함하지 않고 다운한다는 뜻
        분류층 = (Faltten, Dense)레이어 전부다 안 들고옴
    },

    base_model.trainable=False {
        들고온 모델을 학습에서 제외한다는 의미의 코드    
    }

    base_model(x, training=False) = 모델의 모드를 바꾼다
    True 는 훈련 모드
    False 는 추론 모드 이다

}
-->