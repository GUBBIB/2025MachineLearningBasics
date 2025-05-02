# Dropout()
## 하는 일
- **신경망**의 **과대적합(과적합)**을 **방지하기 위한** ``정규화 기법``이다.
- **드롭아웃 레이어**는 **지정된 확률(rate)**만큼 뉴런을 **차단**한다.

## 형식
```python
tf.keras.layers.Dropout(
    rate, seed=None
)
```
*****
- **rate** : 차단할 뉴런의 비율을 정하는 매개변수이다.
    - ``0.0 ~ 1.0`` 사이의 **실수**를 입력받는다.

*****
- **name** : **모델** 안에서 **레이어에 고유 이름을 지정**할 수 있게 해주는 매개변수이다.
    - 주로 **모델 구조를 파악**하거나 **디버깅을 할 때** 유용하다.

*****

## 참고
- [tf.keras.layers.Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) - Dropout 전반