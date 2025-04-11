<!--임시-->
<!--
normalization_layer = layers.Rescaling(1./255)
함수 정의 한거 처럼 normaliztaion_layer 변수를 어딘가의 매개변수로 주면 Rescaling을 쓰겠다는거임
규칙을 정해놓는 거다

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

이런식으로 들어감
-->