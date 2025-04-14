# 코드 설명

## 데이터셋 다운로드
```python
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
```
경로를 통해서 ``flower_photos`` 이미지 세트를 다운하고 ``date_dir`` 변수에 넣었다.
- **사용할 수 있는 이미지**는 약 ``3,6700장``이다.
- 이 이미지 세트에 클래스는 ``daisy``, ``dandelion``, ``roses``, ``sunflowers``, ``tulips``로 총 ``5``개 이다.

## 데이터셋 만들기
```python
batch_size = 32
img_height = 180
img_width = 180
```
- **이미지 크기**는 ``180 x 180`` **batch size**는 ``32``로 설정했다.

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```
```python
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```
- **전체 이미지 데이터** 중 ``20%``는 **검증 데이터**로, ``80%``는 **훈련 데이터**로 사용이 된다.
- 훈련 데이터, 검증 데이터를 구분 하는 방법은 ``subset``의 값으로 구분한다.

## 성능을 높이도록 데이터셋 구성
### 데이터 처리 최적화
```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```
- [cache()](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Dataset/Cache().md) 메소드를 사용해 **한 번 읽은 데이터를 메모리 또는 디스크에 저장**해서 ``Epoch`` **반복 시**에 똑같은 데이터를 다시 로딩하지 않도록 한다.
- [prefetch()](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Dataset/Prefetch().md) 메소드를 사용해 **모델이 학습하는 동안 다음 데이터를 미리 준비하게** 한다.

### 데이터 정규화
```python
normalization_layer = layers.Rescaling(1./255)
```
- 처리하는 데이터가 **이미지 텐서**이기때문에 ``RGB``값인 ``255``로 나누어 **모델이 학습하기 쉬운 형태**인 ``0 ~ 1``사이 **실수**로 **정규화** 시킨다.
- **Rescaling() 함수** 가 ``__call__()`` 메소드를 **오버라이딩** 하고 있기때문에 **Rescaling()**을 참조하는 **normalization_layer 변수**는 ``함수``처럼 사용할 수 있다.

```python
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# 출력
0.0 0.9991613
```
- **정규화**가 잘 되는지 확인하기 위해, 이미지 하나를 꺼내어 ``최소``, ``최대값``을 출력해보면 ``0 ~ 1``사이의 **실수**로 **정규화**가 잘 된것을 확인할 수 있다.

**※ 하지만 훈련에서는 정규화 한 ``normalized_ds`` 변수는 사용하지 않고, 모델 안에서 ``Rescaling 레이어``를 사용한다.**

## 기본 Keras 모델
### 모델 만들기
```python
num_classes = len(class_names)

model = Sequential([
  keras.Input(shape=(img_height, img_width, 3)),
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
```
- 이 모델은 ``Sequential`` 모델로 구성 되었으며, 입력 이미지에 대해서 정규화를 수행한다. 
- 이후 ``Conv2D 레이어``로 **특징 추출**, ``MaxPooling2D 레이어``로 **이미지 크기 축소**를 ``3번`` 진행하며, ``Flatten 레이어``로 **1차원으로 펼친 후**, ``Dense 레이어``로 **최종 분류 결과**를 출력한다.
- **padding**을 ``same``으로 주었기 때문에 가장자리 특징에 대해서 손실은 없다.

### 모델 컴파일
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
- **CNN**의 **기본적인 컴파일 코드**를 사용한다.
자세한건 [Compile() 문서](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Models/Compile().md) 에서 확인할 수 있다.

### 모델 요약
| Layer (type)            | Output Shape         | Param #     |
|-------------------------|----------------------|-------------|
| Rescaling               | (None, 180, 180, 3)  | 0           |
| Conv2D                  | (None, 180, 180, 16) | 448         |
| MaxPooling2D            | (None, 90, 90, 16)   | 0           |
| Conv2D                  | (None, 90, 90, 32)   | 4,640       |
| MaxPooling2D            | (None, 45, 45, 32)   | 0           |
| Conv2D                  | (None, 45, 45, 64)   | 18,496      |
| MaxPooling2D            | (None, 22, 22, 64)   | 0           |
| Flatten                 | (None, 30976)        | 0           |
| Dense                   | (None, 128)          | 3,965,056   |
| Dense                   | (None, 5)            | 645         |

> 총 파라미터 수: **3,989,285개**
- 이 표는 **모델에 포함된** ``레이어의 종류``와 ``형태`` 및 ``학습하는 파라미터 수``를 **요약**해서 보여준다.

## 모델 훈련
```python
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```
- **Epoch**는 ``10``으로 **훈련데이터를** ``10번`` 반복한다.
- **검증 데이터**는 ``전체 꽃 데이터셋(총 3670개)``의 ``20%``를 사용한 ``val_ds 변수``를 넣는다.

## 모델 성능 평가
![첫 번째 훈련 평가](https://github.com/user-attachments/assets/7ceaf13f-18d7-4f76-a232-61ae96d2e544)
```python
Total Epochs: 10

최종 - Accuracy: 0.9911, Validation Accuracy: 0.6621
최종 - Loss: 0.0403, Validation Loss: 1.5442
```

- **모델** 이 ``훈련 데이터``에 대해서는 ``99%``로 거의 완벽하게 맞췄지만, ``검증 데이터`` 에 대해서는 ``66%`` 정도로 성능이 **뚝 떨어졌다**.
- 이는 **모델** 이 ``훈련 데이터`` 를 **암기하듯 외운 상태** 로 새로운 데이터를 **일반화 하지 못한거다**.
- 이런 **과대적합(과적합)** 문제를 해결하기위해 이 챕터에서는 ``데이터 증강`` , ``모델에 드롭아웃 레이어``를 추가 한다.

## 과대적합 해결
### 데이터 증강
- **데이터 증강** 이란 **조금씩 다르게 변형** 해서 **새로운 데이터처럼** 만들어내는 방법이다.
- **과대적합(과적합)** 은 일반적으로 **훈련 예제가 적을 때** 발생한다. 그러므로 ``데이터 증강 기법`` 을 이용해서 기존 예제에서 **추가 훈련 데이터를 생성해** 훈련시킨다.

#### 사용 레이어 설명
[RandomFlip()](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Layers/RandomFlip().md) : **데이터 증강을 위한 레이어**로 이미지를 **무작위**로 ``상하`` 또는 ``좌우``로 뒤집어 학습 데이터를 다양하게 만든다.

[RandomRotation()](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Layers/RandomRotation().md) : **데이터 증강을 위한 레이어**로 이미지를 **무작위**로 ``회전``시켜 학습 데이터를 다양하게 만든다.

[RandomZoom()](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Layers/RandomZoom().md) : 

```python
data_augmentation = Sequential(
  [
    keras.Input(shape=(img_height, img_width, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
```
- 이 모델은 이미지가 들어오면 **무작위**로 ``좌우반전``, ``±18도 범위 내에서 회전``, ``10% 범위 내에서 확대/축소``를 한다.

### 드롭아웃
- **신경망**의 **과대적합(과적합)**을 **방지하기 위한** ``정규화 기법``이다.
- **드롭아웃 레이어**는 **지정된 확률(rate)**만큼 뉴런을 **차단**한다.
- 자세한 설명은 [Dropout 레이어 문서](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/Doc/Layers/Dropout(%EB%89%B4%EB%9F%B0%20%EB%9E%9C%EB%8D%A4%20%EC%B0%A8%EB%8B%A8%20%EB%A0%88%EC%9D%B4%EC%96%B4).md) 에서 확인할 수 있다.

```python
layers.Dropout(0.5) → 50%의 뉴런을 랜덤하게 차단한다.
```

## 모델 재구성
```python
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])
```
- 이전 모델과 달라진 점은 **이미지가 처음 들어올 때**, **데이터 증강 레이어**를 만나면서 ``반전``, ``회전``, ``축소/확대``가 된다는 점과 **마지막에 20%의 랜덤한 뉴런을 차단**한다는 점이다.

## 모델 요약
| Layer                          | Output Shape             | Param #     |
|---------------------------------|--------------------------|-------------|
| `sequential_1 (Sequential)`     | (None, 180, 180, 3)      | 0           |
| `rescaling_2 (Rescaling)`       | (None, 180, 180, 3)      | 0           |
| `conv2d_3 (Conv2D)`             | (None, 180, 180, 16)     | 448         |
| `max_pooling2d_3 (MaxPooling2D)`| (None, 90, 90, 16)       | 0           |
| `conv2d_4 (Conv2D)`             | (None, 90, 90, 32)       | 4,640       |
| `max_pooling2d_4 (MaxPooling2D)`| (None, 45, 45, 32)       | 0           |
| `conv2d_5 (Conv2D)`             | (None, 45, 45, 64)       | 18,496      |
| `max_pooling2d_5 (MaxPooling2D)`| (None, 22, 22, 64)       | 0           |
| `dropout (Dropout)`             | (None, 22, 22, 64)       | 0           |
| `flatten_1 (Flatten)`           | (None, 30976)            | 0           |
| `dense_2 (Dense)`               | (None, 128)              | 3,965,056   |
| `outputs (Dense)`               | (None, 5)                | 645         |

> 총 파라미터 수: **3,989,285개**

- 이전 모델과 학습하는 총 파라미터 수는 같지만 처음 **시작부분**과 **마지막부분**에 ``데이터 증강 모델 Sequentail_1``과 ``Dropout 레이어``가 들어갔다는 점이 다르다.

## 모델 훈련
```python
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
```
- 이번에는 ``Epochs``를 ``15``로 설정하여 충분한 학습 시간을 주게 했다.
- **데이터 증강 모델** 때문에 이미지 데이터가 **매번 달라지므로** 거의 **새로운 이미지로 학습할 수 있는 효과**를 기대할 수 있다.

## 모델 성능 평가
![과적합 문제 해결 후 훈련 성능](https://github.com/user-attachments/assets/1176f360-8d9d-4fa1-a7f4-4a00bc4cd4ac)
```python
Total Epochs: 15

최종 - Accuracy: 0.8941, Validation Accuracy: 0.7643
최종 - Loss: 0.2893, Validation Loss: 0.7948
```

- 이번 훈련에서는 **훈련 데이터**에 대해서는 ``89.41%``의 정확도와 ``0.2893`` 손실값이, **검증 데이터**에 대해서는 ``76.43%``의 정확도와 ``0.7948`` 손실값이 나왔다.
- 이전에는 **훈련 정확도**와 **검증 정확도** 간의 차이가 ``약 33%``로 과적합이 많이 의심 되었으나, 이번에는 ``데이터증강``,``Dropout 레이어`` 추가로 ``약 13%``까지 차이를 줄였다.
- 하지만, **검증 데이터**의 손실값이 여전히 ``0.7948``로 높은편이라, 여전히 일부 입력에 대해 불확실하거나 부정확한 예측을 내리고 있다.