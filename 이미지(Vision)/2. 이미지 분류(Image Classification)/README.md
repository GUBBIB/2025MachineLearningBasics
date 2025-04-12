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


<!--
normalization_layer = layers.Rescaling(1./255)
함수 정의 한거 처럼 normaliztaion_layer 변수를 어딘가의 매개변수로 주면 Rescaling을 쓰겠다는거임
규칙을 정해놓는 거다

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

이런식으로 들어감
-->