# Fit()
## 하는 일
- 모델에 데이터를 넣어서 합습시키는 함수이다.

## 용어 설명
| 용어   | 의미 |
|--------|------|
| **Step**  | 배치 하나를 모델이 처리하는 단위 (`batch_size` 기준) |
| **Batch** | 한 번에 모델에 들어가는 데이터 묶음 (`batch_size`만큼) |
| **Epoch** | 전체 훈련 데이터를 한 바퀴 다 도는 것 |

## 형식
```python
model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose=1,
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
```
*****
- ``x=None`` : ``입력 데이터``를 설정한다.
    - 이 챕터에서는 로드한 데이터인 ``train_images``를 넣는다.

*****
- ``y=None`` : ``입력 데이터``에 대한 ``정답 데이터``를 설정한다.
    - 이 챕터에서는 로드한 데이터인 ``train_labels``를 넣는다.

*****
- ``batch_size=None`` :  한 번의 학습 스텝(step)에서 사용할 데이터 묶음의 크기를 설정한다.
    - **기본값**은 ``None``이지만 TensorFlow가 내부적으로 ``32``를 값으로 준다.

*****
- ``epochs=1`` : ``전체 훈련 데이터``를 **몇 번 반복**해서 학습할지 설정한다.
    - ``epochs`` 가 너무 작으면 **학습이 부족**하고, 너무 크면 **과적합(overfitting)** 이 발생할 수 있다
*****
- ``verbose=1`` : 훈련 중 로그(log) 출력 형태를 설정한다.
    - ``0`` : 아무것도 출력하지 않는다.
    - ``1`` : ``Epochs`` 마다 한 줄씩 출력한다.
        ```python
        Epoch 1/3
        1563/1563 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step - loss: 0.6437 - accuracy: 0.7729
        Epoch 2/3
        1563/1563 ━━━━━━━━━━━━━━━━━━━━ 4s 3ms/step - loss: 0.5001 - accuracy: 0.8200
        ...
        ```

    - ``2`` : 진행 막대 없이 ``Epochs`` 마다 한 줄씩 출력한다.
        ```python
        Epoch 1/3 - loss: 0.6437 - accuracy: 0.7729
        Epoch 2/3 - loss: 0.5001 - accuracy: 0.8200
        Epoch 3/3 - loss: 0.4002 - accuracy: 0.8600
        ...
        ```

*****
- ``callbacks=None`` : 훈련도중 **특정 시점마다 자동**으로 실행되는 **함수의 집합**을 설정한다.

|콜백 이름|설명|
|---------------------|-------------------------------------------|
| `EarlyStopping`| 성능 향상이 없을 경우 **자동으로 훈련 중단** |
| `ModelCheckpoint`| **모델을 저장** (파일로)|
| `ReduceLROnPlateau`| 성능이 정체되면 **학습률 자동 조정**|
| `TensorBoard`| 훈련 로그를 **시각화**|
| `CSVLogger`| 로그를 **CSV 파일로 저장**|


*****
- ``validation_split=0.0`` :

*****
- ``validation_data=None`` :

*****
- ``shuffle=True`` :

*****
- ``class_weight=None`` :

*****
- ``sample_weight=None`` :

*****
- ``initial_epoch=0`` :

*****
- ``steps_per_epoch=None`` :

*****
- ``validation_steps=None`` :

*****
- ``validation_batch_size=None`` :

*****
- ``validation_freq=1`` :

*****
- ``max_queue_size=10`` :

*****
- ``workers=1`` :

*****
- ``use_multiprocessing=False`` :

*****