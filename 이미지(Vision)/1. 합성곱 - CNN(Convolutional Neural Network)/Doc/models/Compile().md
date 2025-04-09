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
- ``loss_weights=None`` : 출력마다 계산되는 손실 함수의 값들에 ``기여 비율(중요도)``을 설정해서 **균형 있는 학습**을 할 수 있게 해준다.
    - **출력**이 ``2개 이상``일 때만 의미가 있으며, 각각의 출력이 [서로 다른 목적을 가질 때](https://github.com/GUBBIB/MachineLearningBasics_TensorFlow/blob/main/%EC%9D%B4%EB%AF%B8%EC%A7%80(Vision)/1.%20%ED%95%A9%EC%84%B1%EA%B3%B1%20-%20CNN(Convolutional%20Neural%20Network)/Doc/models/etc/%EC%84%9C%EB%A1%9C%20%EB%AA%A9%EC%A0%81%EC%9D%B4%20%EB%8B%A4%EB%A5%B8%20%EC%B6%9C%EB%A0%A5%20%EC%98%88%EC%8B%9C.md) 주로 사용한다.

    ※ 출력이 2개 이상이지만 ``loss_weights``를 ``None``으로 주면 각 출력의 **손실**은 동일한 ``가중치(1.0)``로 계산된다.

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