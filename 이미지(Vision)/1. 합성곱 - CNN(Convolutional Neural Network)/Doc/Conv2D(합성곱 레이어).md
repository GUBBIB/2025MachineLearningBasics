# Conv2D

## 하는 일
- 이미지에서 ``특징(feature)`` 을 추출하는 역할을 하며 ``필터(커널)`` 을 이용해 이미지에서 ``중요한 패턴``을 감지한다.<br>
    - 필터를 거쳐서 나온 값이 클수록 특징이 뚜렷하다는 의미이다.

## 형식
```python
tf.keras.layers.Conv2D(
    filters,               # 출력 채널 수 (= 필터 수)
    kernel_size,           # 필터 크기
    strides=(1, 1),        # 필터 이동 간격
    padding='valid',       # 가장자리 처리 방법
    activation=None,       # 활성화 함수
    use_bias=True,         # 바이어스를 쓸지 말지
    kernel_initializer='glorot_uniform',  # 필터 초기값
    bias_initializer='zeros',             # 바이어스 초기값
    input_shape=None       # 첫 레이어일 때 입력 크기
)
```

- ``filters``: 필터(커널)의 개수이며, 보통 ``32``, ``64``, ``128`` 처럼 필터의 수를 점점 늘린다.
    - 필터의 개수가 2의 배수만큼 늘어나면 연산량도 2배 늘어난다.
    - 2의 승수가 아닌 ``10``, ``20``, ``40`` 을 값으로 줄 수도 있다.
    
    ※ 보통은 filters의 개수가 많을수록 여러가지 필터가 있기 때문에 **다양한 특징**을 추출하지만, **무조건적** 으로 그런것은 아니다.

- ``kernel_size`` : **정방향(3)** 또는 **튜플(3, 3)** 등 ``합성곱``을 할 **필터(커널)** 의 크기를 정한다.
    
- ``strides=(1, 1)`` : 필터(커널)가 움직이는 보폭이다.
    - **필터(커널)** 의 크기가 클수록 **출력 크기가 작아지고** **계산량도 줄어**든다. 하지만 ``MaxPooling2D 연산`` 으로 크기를 줄이기 때문에 보통 기본값인 ``(1, 1)``을  사용한다.

- ``padding='valid'`` :
- ``activation=None`` :
- ``use_bias=True`` :
- ``kernel_initializer='glorot_uniform'`` :
- ``bias_initializer='zeros'`` :
- ``input_shape=None`` :