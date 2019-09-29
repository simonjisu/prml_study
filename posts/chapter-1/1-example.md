Example: Polynomial Cureve Fitting
===

제일 간단한 회귀문제를 예시로 든다.

* 실수 입력변수 $$x$$ 로 실수 타겟변수 $$t$$ 를 예측하는 문제다.

# 훈련 데이터 살펴보기

가령 10개의 훈련 데이터를 만들어보는데, 입력변수 $$x$$는 0과 1사이의 실수, 타겟변수 $$t$$ 는 $$\sin(2\pi x)$$ 에서 가우시안 분포에서 샘플링한 작은 노이즈(Noise) 를 줘서 약간의 변형을 가한다.

우리가 알고 싶어하는 $$\sin(2\pi x)$$ 함수는 세상의 진리 혹은 원리라고 생각할 수 있다. 하지만 실제 세상의 개별 데이터는 우리가 알 수 없는 어떤 요소들로 인해서 진실을 알 수 없게 되어있는 경우가 많다. 이를 노이즈가 들어간 데이터로 표현한 것이다. 즉, 쉽게 말하면 입력데이터 $$x$$ 는 함수를 통과해서 정답이 $$t=\sin(2\pi x)$$ 가 나와야하는데, 관측되는 데이터는 항상 어떤 요소에 의해서 조금씩 달라져서 관측된다는 말이다. 

$$\begin{aligned} \textbf{x} & \equiv (x_1, \cdots, x_i, \cdots, x_N)^T, \quad x_i \in [0, 1] \\
\textbf{t} & \equiv (t_1, \cdots, t_i, \cdots, t_N)^T, \quad t_i = \sin(2\pi x_i) + N(\mu, \sigma^2)\end{aligned}$$

이제 우리의 목표는 관측된 훈련데이터들을 사용해서 새로운 입력변수 $$\hat{x}$$ 가 들어왔을 때 타겟변수$$\hat{t}$$ 를 예측하는 것이다.

```python
import numpy as np
import matplotlib.pylab as plt
```

```python
# making data
seed = 62
np.random.seed(seed)
N = 10
x = np.random.rand(N)
t = np.sin(2*np.pi*x) + np.random.randn(N) * 0.1
x_sin = np.linspace(0, 1)
t_sin = np.sin(2*np.pi*x_sin)
plt.plot(x_sin, t_sin, c='green')
plt.scatter(x, t)
plt.xlabel('x', fontsize=16)
plt.ylabel('t', rotation=0, fontsize=16)
plt.show()
```

<img src="https://drive.google.com/uc?id=174y_Vy7ZA0dBMhu8qeWgF_MU17cYM6Yq">

[**확률이론(Probability theory)**](https://ko.wikipedia.org/wiki/%ED%99%95%EB%A5%A0%EB%A1%A0) 은 불확실성을 정확하고 양적인 방식으로 측정할 수 있는 하나의 프레임워크다. [**결정이론(Decision theory)**](https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95_%EC%9D%B4%EB%A1%A0) 은 확률로 표현 것들을 예측(결정) 할 때, 적절한 척도를 가지고 이들을 합리적으로 최적화 하는 이론이다. 

# 다항 함수

불확실성을 해소하기위해 위 두 가지 이론을 사용할 수 있지만, 여기서는 우선 **다항 함수(polynomial fucntion)** 를 통해 접근해볼 수 있다. $$M$$ 은 다항 함수의 **차수(degree)** 라고 하며, 그 식에서 최고의 차수를 가르킨다. 잘 살펴보면, 다항 함수($$y(x, \mathbf{w})$$)은 계수 $$\mathbf{w}$$ 와 연관된 **선형 회귀** 이다. 

$$\tag{1} y(x, \mathbf{w}) = w_0 + w_1 x + w_2 x^2 + \cdots + w_M x^M = \sum_{j=0}^{M} w_j x^j$$

다시 잠깐 정리해서, 지금 하는 것은 세상의 진리($$\sin(2\pi x)$$) 를 모른다고 생각하고 다항 함수를 통해서 이것이 관측된 데이터의 진리가 아닐까 하고 예측해보는 것이라고 할 수 있다. 이 선형 회귀의 계수 $$\mathbf{w}$$ 는 관측된 훈련데이터로 부터 도출할 것이다. 그러면 어떻게 도출할 것인가? 

우리는 다항 함수를 통해 예측된 타겟과 실제 타겟변수의 차이를 구해, 얼만큼 틀렸는지(misfit)를 측정해볼 수 있다. 이를 **목적 함수(object function) / 손실 함수(error/loss function)** 라고 하며, 이 손실함수를 줄임으로써 계수를 구하는 목적을 달성할 수 있다.

여기서는 보통 많이 쓰이는 목적함수로 **MSE(Mean Square Error)** 를 사용한다.

$$\tag{2} E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} (y(x_n, \mathbf{w}) - t_n)^2$$

```python
def error_function(pred, target):
    """MSE function"""
    return (1/2)*((pred-target)**2).sum()
```

# Python Code Solution for Polynomial

우선 $$(N, M+1)$$ 크기의 **방데르몽드 행렬(Vandermode matrix)** 를 정의하고 이를 $$V$$ 라고 한다. 위에서도 말했듯이 $$M$$ 은 다항 함수의 차수(degree) 다.

$$V = \begin{bmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^M  \\
1 & x_2 & x_2^2 & \cdots & x_2^M \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_N & x_N^2 & \cdots & x_N^M
\end{bmatrix}$$

```python
def vandermonde_matrix(x, m):
    """vandermonde matrix"""
    return np.array([x**i for i in range(m+1)]).T
```

3 차 다항 함수의 방데르몽드 행렬을 살펴보자.

```python
M = 3
V = vandermonde_matrix(x, M)
print(V.round(3))

# -----print result-----
# [[1.    0.034 0.001 0.   ]
#  [1.    0.489 0.239 0.117]
#  [1.    0.846 0.716 0.606]
#  [1.    0.411 0.169 0.07 ]
#  [1.    0.631 0.399 0.252]
#  [1.    0.291 0.085 0.025]
#  [1.    0.543 0.295 0.16 ]
#  [1.    0.228 0.052 0.012]
#  [1.    0.24  0.058 0.014]
#  [1.    0.953 0.909 0.867]]
```

이제 행렬로 다항함수 식 **(1)** 을 표현할 수 있게 되는데, 아래와 같다.

$$y = V \cdot w = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{bmatrix} = 
\begin{bmatrix} w_0 + w_1x_1 + w_2x_1^2 + \cdots + w_Mx_1^M \\ w_0 + w_1x_2 + w_2x_2^2 + \cdots + w_Mx_2^M \\ \vdots \\ 
w_0 + w_1x_N + w_2x_N^2 + \cdots + w_Mx_N^M \end{bmatrix}$$

```python
def polynomial_function(x, w, m):
    assert w.size == m+1, "coefficients number must same as M+1"
    V = vandermonde_matrix(x, m)  # shape (x.size, M+1)
    return np.dot(V, w)
```

임의의 계수를 초기화 시키고 다항 함수 값을 살펴본다.

```python
np.random.seed(seed)
w = np.random.randn(M+1)
t_hat = polynomial_function(x, w, M)
print(t_hat.round(3))

# -----print result-----
# [-0.03  -0.208  0.016 -0.2   -0.177 -0.162 -0.204 -0.134 -0.14   0.197]
```

그리고 위에서 정의한 손실 함수를 다시 행렬에 맞게 바꿔보고, 조금 더 간편하게 하기 위해서 **잔차(residual)** $$r=y-V\cdot w$$ 를 정의해서 다시 바꿔본다.

$$
E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^{N} (y(x_n, \mathbf{w}) - t_n)^2 = \frac{1}{2} \Vert y - V \cdot w \Vert^2 = \frac{1}{2} \Vert r \Vert^2
$$

우리 목적은 손실 함수을 최대한 줄여서, 즉 최소값을 구해서 계수를 구할 것이다. $$\hat{w} = {\arg \min}_{w} E(w)$$. 또한, 손실 함수는 2차 함수이기 때문에 1차 미분이 0일 때, 유일한 해가 존재한다. 따라서 미분의 연쇄법칙(chain rule)으로 아래 처럼 미분을 진행 할 수 있다.

$$\begin{aligned} \frac{\partial E}{\partial w} &= \begin{bmatrix} \frac{\partial E}{\partial w_0} \\ \frac{\partial E}{\partial w_1} \\ \vdots \\ \frac{\partial E}{\partial w_M} \end{bmatrix} \\ \\
&= \begin{bmatrix} 
\frac{\partial E}{\partial r_1}\frac{\partial r_1}{\partial w_0} + \frac{\partial E}{\partial r_2}\frac{\partial r_2}{\partial w_0} + \cdots +\frac{\partial E}{\partial r_N}\frac{\partial r_N}{\partial w_0} \\ 
\frac{\partial E}{\partial r_1}\frac{\partial r_1}{\partial w_1} + \frac{\partial E}{\partial r_2}\frac{\partial r_2}{\partial w_1} + \cdots +\frac{\partial E}{\partial r_N}\frac{\partial r_N}{\partial w_1} \\
\vdots \\
\frac{\partial E}{\partial r_1}\frac{\partial r_1}{\partial w_M} + \frac{\partial E}{\partial r_2}\frac{\partial r_2}{\partial w_M} + \cdots +\frac{\partial E}{\partial r_N}\frac{\partial r_N}{\partial w_M} 
\end{bmatrix} \\ \\
&= \begin{bmatrix} 
\frac{\partial r_1}{\partial w_0} & \frac{\partial r_2}{\partial w_0} & \cdots & \frac{\partial r_N}{\partial w_0} \\ 
\frac{\partial r_1}{\partial w_1} & \frac{\partial r_2}{\partial w_1} & \cdots & \frac{\partial r_N}{\partial w_1} \\ 
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial r_1}{\partial w_M} & \frac{\partial r_2}{\partial w_M} & \cdots & \frac{\partial r_N}{\partial w_M} 
\end{bmatrix} \cdot
\begin{bmatrix} \frac{\partial E}{\partial r_1} \\ \frac{\partial E}{\partial r_2} \\ \vdots \\ \frac{\partial E}{\partial r_N} \end{bmatrix} \qquad \cdots (3)\\ \\
&= \frac{\partial r}{\partial w} \cdot \frac{\partial E}{\partial r} \\ \\
&= V^T \cdot (y - V\cdot w) = 0
\end{aligned}$$

**(3)** 번의 식에서 앞쪽에 행렬을 미분해 보면 방데르몽드의 전치 행렬임을 알 수 있다. 

최종적으로 해를 구할 수 있는데, 아래와 같다.

$$w = (V^TV)^{-1}V^Ty$$

```python
def poly_solution(x, t, m):
    V = vandermonde_matrix(x, m)
    return np.linalg.inv(np.dot(V.T, V)).dot(V.T).dot(t)
```

이제 계수를 구해본다.

```python
print(f"Solution of coefficients are {poly_solution(x, t, M).round(3)}")

# -----print result-----
# Solution of coefficients are [ -0.245  11.722 -33.194  21.798]
```

사실 numpy 에서는 더 간편한 기능을 제공하고 있다.

```python
from numpy.polynomial import polynomial as P
print(P.polyfit(x, t, M).round(3))

# -----print result-----
# [ -0.245  11.722 -33.194  21.798]
```

# 최적의 차수(degree) 찾기

최적의 계수를 찾는 문제도 해결되었으니, 이제 우리에게 남은 문제는 최적의 차수를 찾는 것이다. 다항 함수의 차수는 우리가 마음대로 정할 수 있다. 하지만 진리($$\sin(2\pi x)$$)에 가장 가깝게 만드는 최적의 계수는 무엇인가? 이를 찾는 과정을 **모델 비교(model comparison)** 혹은 **모델 선택(model selection)** 이라고 한다. 또한 차수 $$M$$ 처럼사람이 임의적으로 조절할 수 있는 변수를 **하이퍼파라미터(hyperparameter)** 라고 한다.

이전 장([Introduction](https://simonjisu.github.io/prml_study/posts/chapter-1/intro.html))에서 우리는 일반화(generalization) 이 패턴인식의 주요 목적이라고 했다. 좋은 일반화란 얼만큼 진리에 가까운 표현력을 보이는가로 측정할 수 있다. 즉, 여기서는 새로운 데이터가 들어왔을 때, 얼만큼 $$\sin(2\pi x)$$ 에 접근한 가? 를 보면 된다. 그렇다면 이를 어떻게 측정할 것인가?

측정을 위해서 100개의 데이터를 추가로 샘플링해서 새로운 데이터를 만들어 테스트 세트로 구성한다. 

```python
np.random.seed(seed)
N_test = 100 
x_test = np.random.rand(N_test)
t_test = np.sin(2*np.pi*x_test) + np.random.randn(N_test) * 0.1
plt.plot(x_sin, t_sin, c='green')
plt.scatter(x_test, t_test, c='red')
plt.xlabel('x', fontsize=16)
plt.ylabel('t', rotation=0, fontsize=16)
plt.show()
```
<img src="https://drive.google.com/uc?id=1a-lc7wOqcOlBMtecS0w2ZaIXoFrZSUdq">

그리고 매 번 차수($$M$$)를 선택할 때 마다, 훈련 세트에서 최적화된 계수를 구하고, 이 계수를 사용하여 손실 값의 잔차를 훈련 세트와 테스트 세트에 각각 적용해서 구한다. 그 방법 중에 하나는 RMS error(root-mean-sqruare error) 라는 방법으로 구하는데, 식은 아래와 같다. 

$$E_{RMS} = \sqrt{2E(\mathbf{w}^{*})/N}$$

```python
def root_mean_square_error(error, n_samples):
    return np.sqrt(2*error/n_samples)
```

$$\mathbf{w}^{*}$$ 는 $$M$$ 차수에서 최적의 계수, $$N$$ 은 데이터의 갯수다. $$N$$ 을 나눠준 이유는 비교가능도록 크기가 다른 데이터 셋을 동등한 크기로 스케일링 한 것이다. 손실함수가 제곱을 취했기 때문에 예측한 변수와 타겟변수의 차이가 클수록 값이 더 커지는 현상이 있는데, 루트 연산을 취해줌으로써, 타겟 변수와 같은 크기의 스케일로 다시 맞춰진다.

# 오버피팅(over-fitting)

이제 측정할 방법까지 생겼으니 최적의 차수를 골라보자. 차수가 0 부터 9까지 루프문을 돌면서 훈련 세트와 테스트 세트의 RMS error 를 측정해본다.

```python
def get_rms_error(t_hat, t, n_sample, m):
    error = error_function(t_hat, t)
    rms = root_mean_square_error(error, n_sample)
    return rms

all_w = []
all_rms_train = []
all_rms_test = []

for m in range(10):
    optimal_w = poly_solution(x, t, m)
    t_hat = polynomial_function(x, optimal_w, m)
    t_hat_test = polynomial_function(x_test, optimal_w, m)
    
    rms_train = get_rms_error(t_hat, t, N, m)  # N=10
    rms_test = get_rms_error(t_hat_test, t_test, N_test, m)  # N_test = 100
    print(f"M={m} | rms_train: {rms_train:.4f} rms_test: {rms_test:.4f}")
    
    # Plot predicted line
    plt.plot(x_sin, t_sin, c="green", label="sin function")
    plt.plot(x_sin, polynomial_function(x_sin, optimal_w, m), c="red", label=f"model M={m}")
    plt.scatter(x, t)
    plt.xlim((0, 1))
    plt.ylim((-1.25, 1.25))
    plt.xlabel('x', fontsize=16)
    plt.ylabel('t', rotation=0, fontsize=16)
    plt.legend()
    plt.show()
    
    all_w.append(optimal_w)
    all_rms_train.append(rms_train)
    all_rms_test.append(rms_test)
```


그중 차수가 0, 1, 3, 9 인 경우를 살펴본다.

```python
# M=0 | rms_train: 0.6353 rms_test: 0.7221
```

<img src="https://drive.google.com/uc?id=1s1TTlbOGvYsvfLQFs5JMPHMtNNzGi4Bu">

```python
# M=1 | rms_train: 0.4227 rms_test: 0.4508
```

<img src="https://drive.google.com/uc?id=1T7T8vJg2LnyPX5AngcOF_i7_hZuaKwYA">

```python
# M=3 | rms_train: 0.0930 rms_test: 0.1238
```

<img src="https://drive.google.com/uc?id=1lwuWg5WUbJap4ZPOH7twAENyGUhsDHb0">

```python
# M=9 | rms_train: 0.0872 rms_test: 19.2855
```

<img src="https://drive.google.com/uc?id=1ll2uARbNT4zXlr9q9uco3bYae-aGsJJR">

우리의 진리인 $$\sin(2\pi x)$$ 와 가장 가까운 곡선은 $$M=3$$ 일때의 곡선이다. 다른 차수에서는 좋지 않은 표현력(진리 함수와 모양이 비슷하지 않음)을 가지고 있는데, 특히 $$M=9$$ 일 때를 살펴보면 우리의 훈련데이터를 관통하는 아주 정확한 일치성을 보인다. 하지만 곡선을 그려보면 천장과 밑바닥을 뚫는 경우가 발생하는데, 아주 나쁜 표현력을 가지고 있다는 뜻이다. 보통 이러한 현상을 **과대적합(over-fitting)** 이라고 한다. $$M=0$$ 혹은 $$M=1$$ 의 RMS error 가 상대적으로 엄청 크진 않지만, 해당 함수로는 $$\sin$$ 함수를 표현해내기에는 역부족해 보인다. 이러한 현상을 **과소적합(under-fitting)** 이라고 한다.

차수의 선택에 따른 RMS error 을 그려보면 아래와 같다. 

```python
plt.scatter(np.arange(10), all_rms_train, facecolors='none', edgecolors='b')
plt.plot(np.arange(10), all_rms_train, c='b', label='Training')
plt.scatter(np.arange(len(all_rms_test)), all_rms_test, facecolors='none', edgecolors='r')
plt.plot(np.arange(len(all_rms_test)), all_rms_test, c='r', label='Test')
plt.legend()
plt.xlim((-0.1, 10))
plt.ylim((-0.1, 1.2))
plt.ylabel("root-mean-squared Error", fontsize=16)
plt.xlabel("M", fontsize=16)
plt.show()
```

<img src="https://drive.google.com/uc?id=1DiQLWSPY08FlB51AKP69KzIx7tfsYw-K">

훈련 세트와 테스트 세트 간의 RMS error 차이가 처음에는 줄어들다가 나중에는 커지는 것을 알 수 있다. 즉 최적의 차수는 훈련 세트와 테스트 세트 간의 측정척도가 작으며, 테스트 세트에서 어느정도 낮은 측정 척도를 가지고 있어야 한다는 것을 알 수 있다.

이제 각각의 계수를 출력해본다. 차수 $$M$$ 가 커질 수록 계수가 커지는 것을 확인 할 수 있다.

```python
np.set_printoptions(precision=3)
for i in [0, 1, 3, 9]:
    print(f"coefficients at M={i} is {all_w[i]}")
    
# -----print result-----
# coefficients at M=0 is [0.149]
# coefficients at M=1 is [ 0.961 -1.739]
# coefficients at M=3 is [ -0.245  11.722 -33.194  21.798]
# coefficients at M=9 is [-5.400e+01  2.606e+03 -3.763e+04  2.686e+05 -1.111e+06  2.839e+06 -4.546e+06  4.438e+06 -2.411e+06  5.575e+05]
```

## 오버피팅을 피하는 방법

과대적합(over-fitting)을 피하는 방법은 무엇일까? 복잡한 모델일 수록 데이터가 많으면 오버피팅을 피해갈 수 있다. 아래의 예시를 보자. 같은 차수(M=9)의 모델로 훈련 데이터 15 개와 100개의 차이로 학습된 곡선이 달라졌음을 알 수 있다.

```python
np.random.seed(seed)
N1 = 15
N2 = 100 
x1, x2 = np.random.rand(N1), np.random.rand(N2)
t1 = np.sin(2*np.pi*x1) + np.random.randn(N1) * 0.1
t2 = np.sin(2*np.pi*x2) + np.random.randn(N2) * 0.1

optimal_w1 = poly_solution(x1, t1, m=9)
optimal_w2 = poly_solution(x2, t2, m=9)

# Plot predicted line
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

def plot(x, t, x_sin, t_sin, optimal_w, m, ax):
    ax.plot(x_sin, t_sin, c="green", label="sin function")
    ax.plot(x_sin, polynomial_function(x_sin, optimal_w, m), c="red", label=f"model N={len(x)}")
    ax.scatter(x, t)
    ax.set_xlim((0, 1))
    ax.set_ylim((-1.25, 1.25))
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('t', rotation=0, fontsize=16)
    ax.legend()
    
plot(x1, t1, x_sin, t_sin, optimal_w1, m=9, ax=ax1)
plot(x2, t2, x_sin, t_sin, optimal_w2, m=9, ax=ax2)

plt.show()
```

<img src="https://drive.google.com/uc?id=1QNooUihW1_hus_swf6K69zYwVfeR3Ut2">

즉, 데이터가 많아 질 수록 오버피팅 문제는 적어진다. 또 다른 말로 해석하면, 큰 데이터 세트일 수록 더 복잡한(유연한) 모델을 만들 수 있다. 

복잡한 문제를 풀려면, 더 복잡한 모델을 만들어야 한다는 생각이 이쯤되면 생길 것이다. 차후에 우리는 **최대 가능도(mamximum likelihood)** 를 통해서 모델 파라미터(계수)를 찾는 방법을 배울 것이고, 오버피팅 문제또한 최대 가능도의 한 일반적인 특성으로 이해할 것이다.  그리고 **베이지안(Bayesian) 접근법** 을 통해 오버피팅을 해소할 수 있다는 것도 배울 것이다.


## 정규화(Regularization)

위에 방법을 배우기 전에 우선 **정규화(regularization)** 이라는 다른 방법을 알아본다. 정규화는 계수가 더 커지지 않도록 손실 함수에 패널티(penalty)를 더하는 방법이다. **(2)** 번식을 아래와 같이 고쳐본다.

$$\tag{4} E(\mathbf{w}) = \dfrac{1}{2} \Vert y - V \cdot w \Vert^2 + \frac{\lambda}{2} \Vert \mathbf{w} \Vert^2$$

여기서 $$\Vert \mathbf{w} \Vert^2 \equiv \mathbf{w}^T\mathbf{w}=w_0^2 + w_1^2 + \cdots w_M^2$$ 다. **정규화 계수** $$\lambda$$ 는 추가적으로 제약조건의 비중을 조절하는 하이퍼파라미터다. 

여러가지 정규화 방법이 있으나, 여기서는 제일 간단한 계수의 제곱을 손실함수에 더해주는 형식으로 패널티를 더했다. **(4)** 식의 해를 구하는 것은 간단하다. 

$$\begin{aligned}
\frac{\partial E(w)}{\partial w} &= V^Ty-V^TV\cdot w+\lambda w = 0 \\
w &= (V^TV- \lambda I_{(M+1)})^{-1}V^Ty
\end{aligned}$$

```python
def ridge_solution(x, t, m, alpha=0):
    V = vandermonde_matrix(x, m)
    return np.linalg.inv(np.dot(V.T, V) - alpha * np.eye(m+1)).dot(V.T).dot(t)
```

정규화 계수의 효과는 아래의 그림을 보면 극명하다. 정규화 계수가 클수록 계수 $$w$$ 를 강력하게 규제하며 더이상 커지지 못하게 한다. 또한, 그림에서 볼 수 있듯이 모델의 복잡성을 줄여주고 과대적합을 막아준다.

```python
M=9
optimal_w1 = ridge_solution(x, t, m=M, alpha=1e-8)
optimal_w2 = ridge_solution(x, t, m=M, alpha=1.0)

# Plot predicted line
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

def plot_ridge(x, t, x_sin, t_sin, optimal_w, m, text, ax):
    ax.plot(x_sin, t_sin, c="green", label="sin function")
    ax.plot(x_sin, polynomial_function(x_sin, optimal_w, m), c="red", label=f"model M={m}")
    ax.scatter(x, t)
    ax.set_xlim((0, 1))
    ax.set_ylim((-1.25, 1.25))
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('t', rotation=0, fontsize=16)
    ax.legend()
    ax.annotate(text, (0.6, 0.5), fontsize=14)
    
plot_ridge(x, t, x_sin, t_sin, optimal_w1, m=M, text='lambda = 1e-8', ax=ax1)
plot_ridge(x, t, x_sin, t_sin, optimal_w2, m=M, text='lambda = 1.0', ax=ax2)

plt.show()
```

<img src="https://drive.google.com/uc?id=123aWBw7MtpnJ-k2ZpKGkTu_acEDxASMA">


```python
print(f"coefficients at lambda=1e-8 is {optimal_w1.round(3)}")
print(f"coefficients at lambda=1.0 is {optimal_w2.round(3)}")

# -----print result-----
# coefficients at lambda=1e-8 is [  0.104   0.223  33.063 -91.357   3.467  90.523  39.05  -58.172 -72.941 55.889]
# coefficients at lambda=1.0 is  [  0.364   0.321   0.074  -0.155  -0.312  -0.409  -0.465  -0.495  -0.507 -0.51 ]
```

통계학에서 이러한 테크닉을 **수축 방법(shrinkage method)** 이라고 하는데, 그 이유는 계수의 값을 줄여주기 때문이다. 특히, 예시에서 나온 방법은 **ridge regression** 이다. 향후에 이야기할 신경망에서는 **weight decay** 라고도 한다. 그렇다고해서 아주 높은 정규화를 항상 강하게 가져가야하는 것은 아니다. 위에 그림에서 정규화 계수가 1인 모델은 과소적합을 야기했기 때문이다. 

아래 그림은 정규화 계수가 커짐에 따라 RMS error 를 구한 것이다. 차수가 9 임에도 불구하고 낮은 RMS error 를 유지하고 있다.

<img src="https://drive.google.com/uc?id=1UDbMp5Giv6fkTv8ntoiUCqKXRnuF33-X">

---

(책에서 나온 그래프랑 상이 할 수도 있는데, 이는 seed 가 달라서 책에 있는 데이터와 완전히 같을 수가 없기 때문이다.)
