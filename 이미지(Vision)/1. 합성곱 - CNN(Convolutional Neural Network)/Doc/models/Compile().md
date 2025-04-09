# Compile()

## 하는 일
- **모델**을 학습하기 전에, 학습에 필요한 ``방식/전략`` **(옵티마이저, 손실 함수, 평가 지표 등)** 을 설정하는 함수이다.

## 형식
```python
model.compile(
    optimizer='rmsprop',        
    loss=None,                  
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile='auto',
    auto_scale_loss=True
)
```
*****
- ``optimizer='rmsprop'`` : 모델이 **어떻게 학습할지(가중치를 어떻게 조정할지)** **알고리즘**을 결정하는 매개변수이다.
    - 기본값은 ``rmsprop`` 이며, 주로 사용되는 옵티마이저로는 ``adam``, ``sgd``, ``adagrad`` 등이 있다.
    - **CNN**에서는 주로 ``adam`` , **RNN**에서는 ``rmsprop``, **NLP**에서는 ``adagrad`` 또는 ``Ftrl`` 를 사용한다.

*****
- ``loss=None`` : 모델의 ``예측값``과 ``실제 정답`` 간의 차이를 **수치화**해서 나타내는 함수이며, ``옵티마이저``가 **가중치를 조정하는 기준**으로 사용된다.
    - 기본값이 ``None``이지만 손실(loss) 함수를 지정해주지 않으면 오류가나므로 반드시 값을 설정해 줘야 한다.
    - **회귀 문제**에서 주로 사용하는  ``MeanSquaredError``, ``MeanAbsoluteError``와 **다중 클래스 분류 문제**에서 사용하는  ``CategoricalCrossentropy(원 - 핫 인코딩)``, ``SparseCategoricalCrossentropy(정수 라벨)`` 가 있으며 이외에도 ``이진 분류``, ``확률 분포 비교`` 등 여러가지 손실함수가 있다.
    - 이 챕터에서는 다중 클래스 분류를 해결 하는 ``SparseCategoricalCrossentropy`` 함수를 사용한다.

*****
- ``loss_weights=None`` : 

*****
- ``metrics=None`` :

*****
- ``weighted_metrics=None`` : 

*****
- ``run_eagerly=False`` :

*****
- ``steps_per_execution=1`` :

*****
- ``jit_compile='auto'`` :

*****
- ``auto_scale_loss=True`` :

*****