# Flatten()

## 하는 일
- ``Conv2D``, ``MaxPooling2D`` 등을 통해 얻은 ``다차원(2D/3D 등) 특징 데이터``를 ``1차원 벡터``로 변환하는 레이어다.
    - ``Dense 레이어`` 가 **1차원 데이터** 만 입력받기 때문에, 그 전에 ``Flatten 레이어`` 를 사용해 데이터를 1차원으로 변환해야 한다.

## 작동 방식
![Image](https://github.com/user-attachments/assets/82bf12e1-a8ad-452c-a348-cfdf687c1d61)

## 형식
```python
tf.keras.layers.Flatten()
```