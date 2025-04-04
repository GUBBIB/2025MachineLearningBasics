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

- ``filters``: 필터(커널)의 개수이다.<br>
    **※ 보통은 filters의 크기가 클수록 다양한 특징을 추출하지만 무조건적으로 그런것은 아니다.**

- kernel_size: 필터(커널)의 크기이다.