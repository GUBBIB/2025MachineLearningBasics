# 코드 설명

<!-- 
```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()
C:\Windows\System32\myTensorflow\lib\site-packages\keras\src\layers\convolutional\base_conv.py:107: UserWarning: Do not pass an input_shape/input_dim argument to a layer. When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
  ```

합성곱 층 만들기에서 ``model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))`` 이런식으로 첫번째 레이어에 shape을 직접 지정하는 것보다 Import 객체를 사용하는 것이 권장된다고 해서 코드 수정 -->
