# deeplearning-tf

## Tensorflow 를 이용한 Deeplearning 구현

####  pytorch-study-doyeong

# Batch_Normalization

## SGD

딥러닝을 학습하는 알고리즘으로 가장많이 사용되는 방법. 여기서 gradient descent는 파라미터에 의해 정의된 특정 cost함수를 최적화하기 위해 gradient를 이용하여 cost함수가 감소하는 방향으로 파라미터를 학습시켜주는 과정.

- 문제점1 : network학습을 하기 위해 하이퍼파라미터의 초기값 설정을 굉장히 신중하게 해줘야함. -> layer가 깊어질 수록 변화량이 커짐(Covariate Shift)
  - 즉, layer가 쌓일수록 각 layer들은 변화된 입력 분포를 받음
  - 따라서 학습을 원할하게 하기 위해 layer마다 입력 분포를 일정하게 유지시키고 싶음

- 문제점2 : activaiton 함수로 sigmoid를 사용하는데 함수의 특성상 x의 절대값이 커질수록 gradient값이 매우 작아짐
  - x의 분포가 0으로부터 멀어진다면 학습이 잘 진행되지 않음.

## Internal Covariance Shift

- 학습의 불안정화가 일어나는 이유
- 네트워크의 각 레이어나 Activation 마다 입력값의 분산이 달라지는 현상

Covariate Shift

- 이전 레이어의 파라미터 변화로 인해 현재 레이어의 입력의 분포가 바뀜

Internal Covariate Shift

- 레이어를 통과할 때마다 Covariate Shift가 일어나면서 입력의 분포가 변하는 현상

## Whitening

- ICS의 현상을 막기위해 각 레이어의 입력의 분산을 평균 0, 표준편차 1인 입력값으로 정규화 시키는 방법
- But gradient descent step의 효과를 줄임
  - 계산량 많고, 일부 파라미터들의 영향이 무시됨	

## Batch Normalization

- 평균과 분산 조정하는 과정이 별도의 과정으로 떼어진 것이 아니라 신경망 안에 포함되어 학습시 평균과 분산을 조정하는 과정(감마, 베타)

- B = {x1, ... , xm} -> BN = {y1, ..., ym}

- 각각의 feature들을 독립적으로 정규화 하는 방식. 즉, 각각의 feature들의 평균 분산을 구함 d차원 입력이면 x = (x1, ... , xd)에 대해서 각각 정규화
- 각 정규화 한 값에 감마(scale), 베타(shift) 수행

## 효과

- 어떠한 경우에서든 batch norm을 사용했을 때 학습 속도가 매우 빠름
- 배치 정규화를 이용하지 않는 경우에는 초깃값이 잘 분포되어 있지 않으면 학습이 전혀 진행되지 않는 현상도 있음
- 즉, batch normalization을 사용하면 학습이 빨라지며, 가중치 초깃값에 크게 의존하지 않아도됨



## Dropout

- 왜함?
  - Rugularlization(dnn에서 오버피팅 막는것)
  - Small network -> reduce overfit
- 오버피팅이 문제가 되는 경우가 많은데 이는 신경망이 훈련 데이터에 지나치게 적응되어 그 외의 데이터는 제대로 결과를 뽑아내지 못함
  - dropout을 통해 오버피팅 억제
- 뉴런을 임의로 삭제하면서 학습하는 방법
  - small network를 여러번 학습(많은 subnetwork) 
  - 그리고 average함으로써 performance를 올림(Ensemble)
- Train시에 은닉층의 뉴런을 무작위로 삭제(출력을 0으로 설정) -> 신호전달X
  - back propagation할때도 학습X
- Test시에는 모든 뉴런에 신호 전달(test시에는 각 뉴런의 출력에 훈련 때 삭제한 비율을 곱하여 출력)
  -  어떤 뉴런은 자주 켜지고, 어떤 뉴런은 자주 꺼지고 등 균등하지 못한 상황이 발생함
  - 그래서 꺼질 확률에 대한 weight를 뉴런에 각각 곱함으로써(pW)  위의 상황 작게나마 해결해주려함
- 앙상블 기법과 유사

- 문제점
  - Underfitting이 발생할 수 있음(performance가 안나올 수 있음)

## RNN

### 구조

- 음성인식, 자연어 같은 sequence data를 처리

  - 이전의 data가 다음 data에 영향을 끼치는 데이터

- 어떠한 시점을 계산할 때 이전 시점의 연산이 영향을 미침
- $h_t = f_w(h_{t-1}, x_t)$
- $h_t$ : new state, $f_w$ : some function with parameters W, $h_{t-1}$ : old state, $x_t$ : input vector at some time step
- $h_t = tanh(W_{hh}*h_{t-1} + W_{xh}*x_t)$

### 장점

- 이전 정보를 현재의 문제해결에 활용 가능

### 단점

- Long-Term Dependency

  -  길이가 길어짐에 따라 신경망을 통과할 때마다 기울기 값이 조금씩 작아(커)져, t까지 역전파가 되기 전에 0(무한)이 되어 gradient Vanishing(Gradient Exploding) 문제 발생

    

## LSTM

### 구조

- Sequence가 길어졌을 때를 대비해서 RNN의 hidden state에  memory cell 추가

- 어떠한 정보를 forget, remember 하는 메커니즘

- 단순히 한개의 tanh layer가 아닌 4개의 layer가 서로 정보를 주고받는 구조

  1) Cell state
     - 정보가 그대로 흐르도록 하는 역할

  2) Forget gate
     - cell state 에서 sigmoid 를 거쳐 어떤 정보를 잊을지 결정
  3) Input gate
     - 새로운 정보에서 어떤 것을 cell state에 저장할지 결정
     - sigmoid를 거치고 tanh layer를 거쳐 새로운 후보 vector를 생성
  4) Cell state update
     - 버릴 정보와 업데이트 할 정보를 (+)

  5. Output gate
     - 어떤 정보를 output으로 보낼지 결정
     - sigmoid를 거친 Input data * $tanh(C_t)$

### 장점

- cell state가 흐르면서 (forget, input, output)이 점진적으로 변화됨
  - Gradient 문제 해결

### 단점

- output gate 가 $C_t$를 전달하기 때문에 LSTM블록별 cell state는 output에 따라 달라짐
- output gate의 sigmoid가 0이 될 경우 cell state에 접근X



## GRU

### 구조

- LSTM에서 사용하던 Cell state 없이 h사용

  - 이전데이터를 다음 셀에 넘겨주는 역할

  1) Reset gate(rt)를 계산해서 gt를 계산
  2) Update gate(zt)를 통해 h(t-1)과 gt 간의 비중을 결정
  3) zt를 이용해 최종 ht를 계산

### 장점

- Vanishing gradient 해결
- 계산이 더 빠름(lstm에 비해)

### 단점

- LSTM과 비슷

### 정리

- RNN은 non linear activation을 계속 통과해서 vanishing gradient problem
- LSTM이 vanishing gradient problem을 cell state를 통해 해결
  - Non linear activation function을 우회
- cell state를 없애고 lstm보다 좀더 효과적으로 해결

## Attention

- seq2seq의 단점 해결
  - encoder에서 일정 크기로 모든 시퀀스 정보를 압축하여 표현하려고 하기 때문에 정보손실이 발생하는 문제
- decoder가 단순히 encoder의 압축된 정보만을 받아 예측값을 출력하는 것이 아니라, decoder가 출력되는 시점마다 encoder의 전체 입력 문장을 검토
  - encoder에서는 중요한 단어에 대하여 더 큰 가중치를 주어 중요성 나타냄

