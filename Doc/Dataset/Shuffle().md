# Shuffle()
## 하는 일
- 데이터셋에서 데이터를 무작위로 섞는 작업을 한다.

## 형식
```python
train_ds = train_ds.shuffle(buffer_size)
```
- 데이터를 로드할 버퍼의 크기를 값으로 주면 그 값 만큼 데이터를 들고와 섞는다.
- 단, ``buffer_size``의 **크기가 데이터의 크기보다 크거나 같다면** **``완벽하게``** 섞이지만, ``buffer_size``의 **크기가 작다면 부분적으로만** 섞인다.

## 예시
```python
dataset = tf.data.Dataset.from_tensor_slices([[i, i+2, i+4] for i in range(10)])

dataset = dataset.shuffle(buffer_size=2)

for data in dataset:
    print(data.numpy())

[1 3 5]
[0 2 4]
[3 5 7]
[4 6 8]
[5 7 9]
[ 6  8 10]
[2 4 6]
[ 7  9 11]
[ 9 11 13]
[ 8 10 12]
```
- ``[0, 2, 4]`` 부터 ``[9, 11, 13]`` 의 값을 가지는 ``Tensor`` 를 생성하고 ``shuffle``로 섞은 후, 출력을 해보면 부분적으로만 섞인걸 볼 수 있다.

## 참고
- [buffer_size에 따른 shuffle()의 이해](https://velog.io/@godsihyeong/buffersize%EC%97%90-%EB%94%B0%EB%A5%B8-shuffle%EC%9D%98-%EC%9D%B4%ED%95%B4) - shuffle 예시