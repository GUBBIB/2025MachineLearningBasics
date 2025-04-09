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
- ``metrics=None`` :모델이 학습 중 얼마나 잘 예측하고 있는지 **평가**하는 기준을 설정한다.
    - 대표적인 지표로는 **정확도(``accuracy``)**, **평균 절대 오차(``mae``)**, **평균 제곱 오차(``mse``)** 등이 있다.

*****
- ``weighted_metrics=None`` : 출력마다 평가되는 ``metrics``의 값들에 ``가중 평균``을 구한다.
    - 가중 **평균**을 구하는 거기 때문에 출력이 ``2개 이상``일 때만 의미가 있다.
    - 이때의 **가중치**는 ``loss_weights``에 설정한 값을 기준으로 한다.

    ※ ``가중 평균`` 이란 특정 값에 ``가중치``를 곱해서 **평균**을 내는 방법이다.

*****
- ``run_eagerly=False`` : 모델이 학습할 때, 내부 코드를 어떤 방식으로 실행할지 정한다.
    - ``그래프 실행 방식``과 ``Eager 실행 방식``이 있다.
    - ``그래프 실행 방식(False)``은 **컴파일러** 처럼 한번에 계산하여 그래프를 **생성후**에 실행한다. 
        - 훈련 속도가 빠르지만 디버깅이 어렵다.
    - ``Eager 실행 방식(True)``은 **인터프리터** 처럼 한줄 씩 계산한다음 실행한다. 
        - 훈련 속도는 느리지만 디버깅이 쉽다.

*****
- ``steps_per_execution=1`` : 훈련 중 모델이 ``몇 스텝``마다 그래프를 **한 번씩 실행**할지 결정한다.<br>
    - ``Step``, ``Batch``, ``Epoch``<br>

    | 용어   | 의미 |
    |--------|------|
    | **Step**  | 배치 하나를 모델이 처리하는 단위 (`batch_size` 기준) |
    | **Batch** | 한 번에 모델에 들어가는 데이터 묶음 (`batch_size`만큼) |
    | **Epoch** | 전체 훈련 데이터를 한 바퀴 다 도는 것 |


*****
- ``jit_compile='auto'`` : 모델 훈련 시 **XLA(Accelerated Linear Algebra)**라는 컴파일러를 사용할지 자동으로 결정할지 정한다.
    - ``Auto``
        - ``TensorFlow``가 **자동으로 판단**해서 ``XLA`` 사용여부를 결정한다. ``(기본값)``
    - ``True``
        - XLA을 켜서 JIT 컴파일을 실행한다. 속도는 빨라지지만 **오류가 발생할 가능성**이 높아진다.
    - ``False``
        - XLA 컴파일 방식을 사용하지 않는다.

        **!** ``XLA 컴파일러`` 는 **실행방식**이 ``그래프`` 일 때만 작동하고, 계산 그래프를 **더 깊이 최적화하는 방식**이다.<br>
        **!!** 만약 모델 실행 방식이 ``Eager``라면 ``jit_compile``은 의미가 없다.

*****
- ``auto_scale_loss=True`` :

*****
## 예시
```python
# 훈련 방식
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```
```python
# 마지막 훈련 시점 출력 결과
Epoch 10/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - accuracy: 0.7729 - loss: 0.6437 - val_accuracy: 0.7071 - val_loss: 0.869
```

! ``loss``는 모델이 **얼마나 틀렸는지**, ``accuracy``는 모델이 **얼마나 맞췄는지** ``비율``로 측정한 값이다.