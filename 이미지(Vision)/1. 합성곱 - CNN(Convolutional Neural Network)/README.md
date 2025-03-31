# 코드 설명

## keras
### keras란?
**keras**는 ``Python`` 으로 구현된 **쉽고 간결한 딥러닝 라이브러리**이다.

### keras를 사용하는 이유
1. 모듈화가 가능하다.
2. 간결하며 이해하기가 쉽다.
3. 새로운 클래스나 함수로 모듈을 쉽게 확장할 수 있다.

### keras 라이브러리의 모듈
이 챕터에서 사용하는 모듈은 ``datasets``, ``layers``, ``models`` 3가지이다.

- **dataset** : 딥러닝 모델을 학습시키기 위한 데이터셋을 제공하는 모듈로 ``CIFAR10``, ``MINIST`` 가 있다.<br>
이 챕터에서는 ``datasets.cifar10.load_data()``를 통해서 ``CIFAR10`` 데이터셋을 로드한다.

※ ``CIFAR10`` 데이터셋은 10개의 **클래스/객체** (자동차, 비행기, 고양이, 강아지 등)로 분류된 ``32x32`` 크기의 **컬러(RGB)** 이미지 데이터셋이다.

- **layers** : 

## 참고
- [[Keras & Tensorfow] 개념](https://golikeriver97.tistory.com/41)

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
