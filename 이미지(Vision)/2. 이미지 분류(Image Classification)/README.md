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
**정규화**가 잘 되었는지 확인하기 위해, 이미지 하나를 꺼내어 ``최소``, ``최대값``을 출력해보면 ``0 ~ 1``사이의 **실수**로 **정규화**가 잘 된것을 확인할 수 있다.

<!--
normalization_layer = layers.Rescaling(1./255)
함수 정의 한거 처럼 normaliztaion_layer 변수를 어딘가의 매개변수로 주면 Rescaling을 쓰겠다는거임
규칙을 정해놓는 거다

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

이런식으로 들어감
-->