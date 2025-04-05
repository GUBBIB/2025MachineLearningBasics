# MaxPooling2D()

## 하는 일
- 주로 ``Conv2D 레이어`` 뒤에 사용되며, 추출한 ``특징``은 **유지하고** 크기는 **줄이는 함수**이다.
- ``Conv2D 레이어``에서 계산된 결과에서 ``풀링(Pooling) 영역[ex: 2x2]``을 나눠, 그 안에서 **가장 큰 값**을 선택하는 방식이다.

    - 크기를 줄이기 때문에 **계산량**도 줄고, **과적합** 도 막아준다.

## 작동 방식 예시 이미지
<img width="640" alt="Image" src="https://github.com/user-attachments/assets/2c9f1412-0f3b-4ec0-9e69-9bffcd941b1e" />

## 형식
```python
tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),   # 풀링 영역의 크기
    strides=None,       # 이동 간격 (기본은 pool_size와 같음)
    padding='valid'     # 가장자리 처리 방식
)
```
*****
- ``pool_size=(2, 2)`` : ``풀링(pooling)영역``의 크기를 정한다.
    - 보통 ``(2, 2)``정도의 크기를 사용해서 이미지의 크기를 **절반으로** 줄인다.

*****
- ``strides=none`` :

*****
- ``padding='valid'`` :

*****

## 참고
- [폴링 계층, Pooling Layer](https://dsbook.tistory.com/79) - 풀링 예시 이미지