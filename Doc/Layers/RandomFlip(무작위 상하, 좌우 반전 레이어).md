# RandomFlip()
## 하는 일
- **데이터 증강을 위한 레이어**로 이미지를 **무작위**로 ``상하`` 또는 ``좌우``로 뒤집어 학습 데이터를 다양하게 만든다.

## 형식
```python
tf.keras.layers.RandomFlip(
    mode="horizontal", 
    seed=None, 
    name=None
)
```
*****
- **mode** : 기본값은 ``horizontal(좌우 반전)``이며, 이외에도 ``vertical(상하 반전)``, ``horizontal_and_vertical(좌우, 상하 반전)``이 있다.

*****
- **seed** : ``RandomFlip`` 레이어의 결과값은 무작위로 나오지만 **항상 같은 결과가 나오게** 할 수 있는 매개변수이다.<br>
    - 주로 디버깅을 할 때 사용한다.

*****
- **name** : **모델** 안에서 **레이어에 고유 이름을 지정**할 수 있게 해주는 매개변수이다.
    - 주로 **모델 구조를 파악**하거나 **디버깅을 할 때** 유용하다.