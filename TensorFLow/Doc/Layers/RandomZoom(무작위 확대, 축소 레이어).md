# RandomZoom()

## 하는 일
- **데이터 증강을 위한 레이어**로 이미지를 **수직/수평 방향**으로 **무작위**로 ``확대/축소``한다.

## 형식
```python
tf.keras.layers.RandomZoom(
    height_factor,
    width_factor=None,
    fill_mode='reflect',
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    data_format=None,
)
```

