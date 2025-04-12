# Prefetch()
## 하는 일
- 모델이 **훈련하는 동안** ``다음 데이터``를 **미리 준비해두는** 함수이다.

## 형식
```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
```
- ``tf.data.AUTOTUNE``은 **``tf.data`` API** 에서 제공하는 상수이다. 
- **TensorFlow**가 ``시스템 리소스``를 **기반**으로 **최적의 버퍼 크기**를 ``자동``으로 결정한다.