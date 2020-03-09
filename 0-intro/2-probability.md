# 1.2 Probability Theory

패턴인식에서 중요한 개념 중 하나는 **불확실성\(uncertainty\)** 이다. **확률 이론\(Probability Theory\)** 은 불확실성을 정확하고 양적인 방식으로 측정할 수 있는 일관된 프레임워크를 제공해준다. 또한 **결정 이론\(Decision Theory\)** 와 결합하면 현재 가진 정보내에서 최적의 예측을 내릴수 있도록 도와준다.

이 책에서는 다음 예제로 확률을 소개하려고 한다. 확률 변수\(Random Variable\) $$B$$로 `그림 1.2.1`의 박스를 표현한다. 이 확률 변수 $$B$$는 빨간색\($$r$$\)과 파랑색\($$b$$\) 두 가지 경우가 있다. 박스 안에 있는 과일의 종류 또한 확률 변수 $$F$$로 표현하며, 사과\($$a$$\)와 오렌지\($$o$$\) 두 가지 경우가 있다.

![1.2.1](../.gitbook/assets/1.2.1-f1.9.png)

시작하기 전에 사건의 발생 횟수를 총 시행횟수로 나눈 값을 어떤 사건\(event\)의 확률로 정의 한다. 따라서 다음 사건들의 확률을 정의 할 수 있다\(빨강색 박스를 선택할 확률은 40%, 파랑색은 60%다\).

$$\begin{aligned}p(B)&=\dfrac{4}{10} \\ p(R)&=\dfrac{6}{10}\end{aligned}$$

위 정의에 따르면, 확률은 항상 0과 1사이의 값을 가진다. 또한, 상호 배타적\(mutually exclusive\)이거나 모든 결과\(outcomes\)를 포함하는 경우, 모든 확률의 합은 1이 되어야 한다.

여기서 잠깐 확률에서 **합의 법칙\(sum rule\)**과 **곱의 법칙\(product rule\)** 알아보고 온다.

![1.2.2](../.gitbook/assets/1.2.2-f1.10.png)

`그림 1.2.2`에서 $$X$$, $$Y$$ 두 개의 확률 변수가 있다. $$X$$는 $$x_i$$값을 취할 수 있고\($$i$$는 $$(1, \cdots, M)$$까지\), $$Y$$는 $$y_j$$값을 취할 수 있다\($$j$$는 $$(1, \cdots, N)$$까지\). 또한, $$X$$와 $$Y$$에서 표본을 추출하는데 총 시도횟수를 $$N$$이라고 한다. 그리고 $$X$$가 $$x_i$$값을 취하고 $$Y$$가 $$y_j$$값을 취했을 때의 시도 갯수를 $$n_{ij}$$ 라고 한다. 이때 확률은 $$p(X=x_i, Y=y_j)$$라고 하며, $$X=x_i, Y=y_j$$의 **결합 확률\(joint probability\)** 이라고 한다.

실제로 임의로 횟수를 지정해서 계산을 해보자.

![1.2.3](../.gitbook/assets/1.2.3-probexample.png)

```python
np.random.seed(777)
A = np.random.randint(1, 10, size=(3, 5))
fig, ax = plt.subplots(1, 1)
ax.matshow(table, cmap="coolwarm")

for (i, j), z in np.ndenumerate(A):
    ax.text(j, i, f"{z}", ha="center", va="center")
ax.set_xticklabels(np.arange(0, 6))
ax.set_yticklabels(np.arange(0, 4))
ax.set_xlabel("$X$", fontsize=20)
ax.set_ylabel("$Y$", fontsize=20).set_rotation(0)

plt.show()
```

{% tabs %}

$$\tag{1} p(X=x_i, Y=y_j) = \dfrac{n_{ij}}{N}$$

```python
def joint_probability(i, j, A):
    return round(A.T[i, j] / A.size, 4)

joint_probability(0, 1, A)
```

확률 변수 $$Y$$에 관계없이 $$X=x_i$$의 시도 횟수를 $$c_i$$, $$X$$에 관계없이 $$Y=y_j$$의 시도 횟수를 $$r_j$$라고 하면, 다음과 같이 표현할 수 있다.

{% tabs %}

{% tabs %}
{% tab title="title1" %}
$$
a = b
$$
{% endtab %}

{% tab title="title2" %}
```text
python
```
{% endtab %}
{% endtabs %}


