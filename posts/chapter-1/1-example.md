# Example: Polynomial Cureve Fitting

제일 간단한 회귀문제를 예시로 든다.

* 실수 입력변수 $$x$$ 로 실수 타겟변수 $$t$$ 를 예측하는 문제다.

가령 10개의 훈련 데이터를 만들어보는데, 입력변수 $$x$$는 0과 1사이의 실수, 타겟변수 $$t$$ 는 $$\sin(2\pi x)$$ 에서 가우시안 분포에서 샘플링한 작은 노이즈(Noise) 를 줘서 약간의 변형을 가한다.

우리가 알고 싶어하는 $$\sin(2\pi x)$$ 함수는 세상의 진리 혹은 원리라고 생각할 수 있다. 하지만 실제 세상의 개별 데이터는 우리가 알 수 없는 어떤 요소들로 인해서 진실을 알 수 없게 되어있는 경우가 많다. 이를 노이즈가 들어간 데이터로 표현한 것이다. 즉, 쉽게 말하면 입력데이터 $$x$$ 는 함수를 통과해서 정답이 $$t=\sin(2\pi x)$$ 가 나와야하는데, 관측되는 데이터는 항상 어떤 요소에 의해서 조금씩 달라져서 관측된다는 말이다. 

$$
\begin{aligned} \textbf{x} & \equiv (x_1, \cdots, x_i, \cdots, x_N)^T, \quad x_i \in [0, 1] \\
\textbf{t} & \equiv (t_1, \cdots, t_i, \cdots, t_N)^T, \quad t_i = \sin(2\pi x_i) + N(\mu, \sigma^2)\end{aligned}
$$

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

![png](1-example_files/1-example_2_0.png)
