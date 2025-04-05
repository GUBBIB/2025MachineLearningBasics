# MaxPooling2D()

## 하는 일
- 주로 ``Conv2D 레이어`` 뒤에 사용되며, 추출한 ``특징``은 **유지하고** 크기는 **줄이는 함수**이다.
- ``Conv2D 레이어``에서 계산된 결과에서 ``풀링(Pooling) 영역[ex: 2x2]``을 나눠, 그 안에서 **가장 큰 값**을 선택하는 방식이다.

    - 크기를 줄이기 때문에 **계산량**도 줄고, **과적합** 도 막아준다.

![Image](https://github.com/user-attachments/assets/1a8fcd21-2169-4f9c-a66a-e2db5ee76901)


## 형식
```python
tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),   # 풀링 영역의 크기
    strides=None,       # 이동 간격 (기본은 pool_size와 같음)
    padding='valid'     # 가장자리 처리 방식
)
```
*****
- ``pool_size=(2, 2)`` : 

*****
- ``strides=none`` :

*****
- ``padding='valid'`` :

*****