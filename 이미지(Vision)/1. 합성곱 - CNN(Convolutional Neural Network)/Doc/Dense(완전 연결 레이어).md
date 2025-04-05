# Dense()
## 하는 일
-  ``Conv2D``, ``MaxPooling2D`` 로 추출한 ``특징``들을 하나로 모아서, **최종적인 판단(분류, 회귀 등)**이나 **예측**을 내리는 함수이다.

예를들어 ``CNN``에서 **고양이/강아지**를 구분하는 문제에서 ``Dense 레이어`` 가 **특징**들을 보고 **"이건 고양이다!"** 라고 ``결정``을 내리는 부분이다.

## 형식
```python
tf.keras.layers.Dense(
    units,                 # 출력 노드 수
    activation=None,       # 활성화 함수
    use_bias=True,         # 바이어스 사용 여부
    kernel_initializer='glorot_uniform',  # 가중치 초기값
    bias_initializer='zeros',             # 바이어스 초기값
)
```
