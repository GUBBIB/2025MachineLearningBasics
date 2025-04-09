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
- ``optimizer`` :

*****
- ``loss=None`` :

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