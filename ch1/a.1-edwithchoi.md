# a.1 From Set Theory to Probability Theory

## Set Theory

> 최성준님의 [베이지안 딥러닝](https://www.edwith.org/bayesiandeeplearning/) 자료를 많이 참고했습니다.

집합론\(set theory\)은 추상적 대상들의 모임인 집합을 연구하는 수학 이론이다. 기본적인 개념은 위키링크를 달아 두었다.

* [집합\(set\)](https://ko.wikipedia.org/wiki/집합): 특정 조건에 맞는 원소들의 모임
* [원소\(element\)](https://ko.wikipedia.org/wiki/원소_%28수학%29%20): 집합을 이루는 개체, 원소 $$a$$가 집합 $$A$$에 속할 경우 $$a \in A$$라고 표기한다.
* [부분 집합\(subset\)](https://ko.wikipedia.org/wiki/부분집합): 집합 A의 모든 원소가 다른 집합 B에도 속하는 관계일 경우, A는 B의 "부분 집합"이라고 한다.
* [전체집합\(universal set\)](https://ko.wikipedia.org/wiki/전체집합): 모든 대상\(자기 자신도 포함\)을 원소로 포함하는 집합
* [집합의 연산\(set operations\)](https://en.wikipedia.org/wiki/Set_%28mathematics%29#Basic_operations)
  * [합집합\(Unions\)](https://ko.wikipedia.org/wiki/합집합)
  * [교집합\(Intersections\)](https://ko.wikipedia.org/wiki/교집합)
  * [여집합\(Complements\)](https://ko.wikipedia.org/wiki/여집합)
  * [곱집합\(product set, Cartesian product\)](https://ko.wikipedia.org/wiki/곱집합): 각 집합의 원소를 각 선분으로 하는 튜플\(tuple\)들의 집합

    $$A \times B = \{ (a, b): \mathtt{a} \in A, \mathtt{b} \in B\}$$

    * 예시: $$A = \{ 1, 2 \}, B = \{ 3, 4, 5 \} \rightarrow A \times B = \{ (1,3), (1,4), (1,5), (2,3), (2,4), (2,5) \}$$
* [서로소 집합\(disjoint set\)](https://ko.wikipedia.org/wiki/서로소_집합): 공통 원소가 없는 두 집합, $$A \cap B = \emptyset$$
* [집합의 분할\(partition of a set\)](https://ko.wikipedia.org/wiki/집합의_분할): 집합의 원소들을 비공 부분 집합들에게 나눠주어, 모든 원소가 각자 정확히 하나의 부분 집합에 속하게끔 하는 것
  * 예시: $$A = \{ 1, 2, 3, 4 \} \rightarrow \text{partition of set A} = \{ \{1, 2\}, \{3\}, \{4\} \}$$
* [멱잡합\(power set of set A, $$\coloneqq 2^A$$\)](https://ko.wikipedia.org/wiki/멱집합): 주어진 집합의 모든 부분 집합들로 구성된 집합\(the set of all the subsets\)
  * 예시: $$A = \{ 1, 2, 3 \} \rightarrow \text{power set of 2}^A = \{ \emptyset, \{1\},\{2\},\{3\},\{1,2\},\{2,3\},\{1,3\},\{1,2,3\} \}$$
* [집합의 크기\(Cardinality\)](https://ko.wikipedia.org/wiki/집합의_크기): 집합의 "원소 개수"에 대한 척도, $$\vert A \vert$$로 표기 한다. 집합의 크기를 표현하는 용어로 finite, infinite, countable, uncountable, denumerable\(countably infinite\)가 있다.
  * [가산 집합\(countable set\)](https://ko.wikipedia.org/wiki/가산_집합): 관심있는 집합과 자연수의 집합으로 [일대일 함수](https://ko.wikipedia.org/wiki/단사_함수)\(one-to-one function\)관계가 존재하면, 그 집합은 가산 집합이다. 특히, 자연수, 정수, 유리수와 같이 셀수 있는 무한 집합의 경우, 가산 무한\(countable infinite\)이나 가부번 집합\(denumerable set\)이라고 한다.
  * 비가산 집합\(uncountable set\): 가산 집합이 아닌 집합, 실수는 비가산 집합

## Function

![1.2.0.1](./figs/chapter-1/1.2.0.1-function.png)

* [함수/사상\(function/mapping\)](https://ko.wikipedia.org/wiki/함수): 첫 번째 집합의 임의의 한 원소를 두 번째 집합의 오직 한 원소에 대응시키는 이항 관계이다. 입력이 되는 집합 $$U$$를 정의역\(domain\), 출력으로 대응되는 집합 $$V$$를 공역\(codomain\)이라고 한다.

  $$f: \underset{domain}{U} \rightarrow \underset{codomain}{V}$$

* [상\(image\)](https://ko.wikipedia.org/wiki/상_%28수학%29): domain의 원소\(혹은 부분 집합\)가 대응하는 codomain의 원소\(혹은 집합\)

  $$f(x) \in V, x \in U \quad \text{or} \quad f(A) = \{ f(x) \vert x \in A \} \subseteq V, A \subseteq U$$

  반대로 codomain의 원소에 대응하는 domain의 원소를 역상\(inverse image\)이라고 한다\(원소의 역상은 부분 집합이라는 것을 주의\).

  $$f^{-1}(y) = \{ x \in U \vert f(x) \in V \} \subseteq V \quad \text{or} \quad f^{-1}(B) = \{ x \vert f(x) \in B \} \subseteq U, B \subseteq V$$

* [치역\(range\)](https://ko.wikipedia.org/wiki/치역): 함수의 모든 출력값의 집합, 치역은 공역\(codomain\)의 부분 집합이다.

![1.2.0.2](./figs/chapter-1/1.2.0.2-invertible.png)

* [일대일 함수/단사 함수\(one-to-one/injective\)](https://ko.wikipedia.org/wiki/단사_함수): domain의 서로 다른 원소를 codimain의 서로 다른 원소로 대응시키는 함수
* [위로의 함수/전사 함수\(onto/surjective\)](https://ko.wikipedia.org/wiki/전사_함수): domain과 range가 일치하는 함수
* one-to-one 조건과 onto 조건을 모두 만족하면 가역 함수\(invertible function\)라고 한다.

## Measure Theory

[측도\(measure\)](https://ko.wikipedia.org/wiki/측도) 이란 특정 부분 집합에 대해 일종의 "크기"를 부여하며, 그 크기를 가산개로 쪼개어 게산할 수 있게 하는 함수다. 측도가 부여된 집합을 측도 공간\(measure space\)라고 하며, 이를 연구하는 수학 분야를 측도론\(measure theory\)라고 한다.

기본적으로 전체집합\(universial set\) $$U$$가 주어졌을 때, 측도\(measure\)는 $$U$$의 부분집합\(subset\)에 비음수인 실수를 할당한다. 우선 명확히 measure를 정의하기 위해서 필요한 것들을 정의해본다.

* [set function](https://en.wikipedia.org/wiki/Set_function): 집합\(set\)에 대해 어떤 숫자를 부여하는 함수\(ex, cardinality, length, area\), 즉 입력을 집합, 출력은 숫자가 되는 함수
* [$$\sigma$$-field $$\mathcal{B}$$](https://en.wikipedia.org/wiki/%CE%A3-algebra): 다음과 같은 조건을 만족하는 전체집합 $$U$$의 부분 집합 모음$$\mathcal{B}$$를 $$\sigma$$-field 라고 한다\($$\sigma-\text{algebra}$$와 같은 말\). 
  1. $$\emptyset \in \mathcal{B}$$, empty set is included
  2. $$B \in \mathcal{B} \Rightarrow B^{c} \in \mathcal{B}$$, closed under set complement
  3. $$B_i \in \mathcal{B} \Rightarrow \bigcup_{i=1}^{\infty}B_i \in \mathcal{B}$$, closed under countable union
* $$\sigma$$-field는 measure를 부여할 수 있는 최소 단위가 된다. 만약 어떤 원소가 $$\sigma$$-field에 존재하지 않는다면, 그 원소는 측정할 수 없다.
* $$\sigma$$-field 특성
  1. $$U \in \mathcal{B}$$
  2. $$B_i \in \mathcal{B} \Rightarrow \bigcap_{i=1}^{\infty}B_i \in \mathcal{B}$$, closed under countable intersection
  3. $$2^U$$, power set of U 는 가장 단위가 자잘자잘 하게 만든 $$\sigma$$-field
  4. $$\mathcal{B}$$ 는 유한하거나 비가산 둘 중 하나다, 가산 무한/가번부\(countable infinite/denumerable\)가 될 수 없다.
  5. $$\mathcal{B}, \mathcal{C} \text{ are } \sigma \text{-field} \Rightarrow \mathcal{B} \cap \mathcal{C} \text{ is } \sigma \text{-field, but } \mathcal{B} \cup \mathcal{C} \text{ is not}$$
* [가측 공간\(measurable space\)](https://ko.wikipedia.org/wiki/가측_공간): 간단히 말해서, 어떤 집합 $$U$$가 있고 그 집합의 부분집합으로 만들어진 $$\sigma$$-field에 measure를 부여할 수 있는 공간 $$(U, \mathcal{B})$$

[측도\(measure\)](https://en.wikipedia.org/wiki/Measure_%28mathematics%29)를 정의하기 위한 준비는 다 되었다. 정의를 하면 다음과 같다.

* measure $$\mu$$는 가측 공간\(measureable space\)-$$(U, \mathcal{B})$$에서 정의된 set function, $$\mu: \mathcal{B}\rightarrow [0, \infty]$$ 이다.
  1. $$\mu(\emptyset) = 0$$
  2. For disjoint $$B_i$$ and $$B_j \Rightarrow \mu(\bigcup_{i=1}^{\infty}B_i) = \sum_{i=1}^{\infty} \mu(B_i)$$, countable addivitity 
* 즉, 가측 공간\(measurable space\)-$$(U, \mathcal{B})$$과 measure $$\mu$$가 하나의 측도 공간\(measure space\)-$$(U, \mathcal{B}, \mu)$$ 를 구성하게 된다.

## Probability Theory

![1.2.0.2](./figs/chapter-1/1.2.0.3-probexam.png)

`그림 1.2.0.2`에서 $$\Omega$$는 표본 공간\(sample space\)이라고 한다. 표본 공간에서 정의되는 측도\(measure\)는 대문자 P로 작성한다. 무슨 뜻인지는 다음을 계속 읽어본다.

* 확률을 이야가 하기 위해서는 임의적 실험\(random experiment\)를 잘 정의 해야한다.
* [결과\(outcomes\)](https://en.wikipedia.org/wiki/Outcome_%28probability%29)는 임의적 실험에서 발생하며 더이상 나눌수 없는 모든 가능성 있는 현상들을 일컫는 말이다.
* [사건\(event\)](https://en.wikipedia.org/wiki/Event_%28probability_theory%29)은 확률이 부여된 임의적 실험에서 발생한 결과\(outcomes\)의 집합이며, 표본 공간\(sample space\)의 부분 집합이다. 
* [표본\(sample point\)](a1-edwithchoi.md) $$w$$는 표본 공간\(sample space\)에서 임의적 실험을 통해 나올 수 있는 결과\(outcome\)를 말한다.
* [표본 공간\(sample space\) $$\Omega$$](https://ko.wikipedia.org/wiki/표본_공간)은 모든 sample point 의 집합이다. 
* 예를 들어, 공정한 정육면체 주사위를 랜덤으로 던지는 실험이 있다\(**random experiment**\). 결과\(**outcomes**\)로 한 면에 1~6까지 숫자가 보인다. 7은 나올 수 없기 때문에 관찰 가능한 결과\(outcome\)이 아니다. `그림 1.2.0.2` 의 각 점들로 표현되어 있다. 이 그림은 모든 점들이 표본 공간 $$\Omega$$ 내에 정의 되어 있음으로, 모든 점들은 sample point이자 이 임의적 실험의 결과라고 할 수 있다. 마지막으로 "주사위를 굴렸을 때, 보이는 면이 짝수 인 경우", 즉 A로 표기된 $$\Omega$$의 부분 집합은 사건\(**event**\)이다.

이제 확률의 명확한 정의를 내려본다.

* 확률 $$P$$ 는 가측 공간\(measureable space\)-$$(\Omega, \mathcal{A})$$ 에서 정의되는 set function $$P : \mathcal{A} \rightarrow [0, 1]$$ 인데 다음 조건을 만족한다\(기호가 약간 다른데, $$\mathcal{A}$$는 $$\sigma$$-field, 일반 대문자 $$A$$는 $$\sigma$$-field의 부분 집합임으로 잘 구분해야 함\).
  1. $$P(\emptyset) = 0$$
  2. $$P(A) \geq 0, \forall A \subseteq \Omega$$
  3. For disjoint sets $$A_i$$ and $$A_j \Rightarrow \mu(\bigcup_{i=1}^{\infty}B_i) = \sum_{i=1}^{\infty} \mu(B_i)$$, countable addivitity
  4. $$P(\Omega) = 1$$
* 사실상 측도의 정의에서 2, 4번 항목이 추가된 것이다. 즉, 확률은 표본 공간에서 정의된 측도\(measure\) 혹은 set function 이라고 할 수 있겠다. 

지금까지 확률은 가측 공간에서 정의된 것이다. 그렇다면 어떤 사건 $$A$$에 어떻게 확률을 부여할까? 해답은 다음과 같다. 임의적 실험에서 나온 결과로 구성된 표본 공간 $$\Omega$$가 있고, 그 표본 공간에서 발생한 사건 $$A$$에 해당하는 확률을 부여한다. 여기서 확률 할당 함수\(probability allocation function\)이 등장한다.

* [probability allocation function](https://en.wikipedia.org/wiki/Probability_distribution_function)
  * probability mass function: 이산\(discrete\) 표본 공간 $$\Omega$$일 때, $$p: \Omega \rightarrow [0, 1]$$ such that $$\sum_{w\in \Omega} p(w)=1$$ and $$P(A) = \sum_{w \in A} p(w)$$
  * probability density function: 연속\(continuous\) 표본 공간 $$\Omega$$일 때, $$p: \Omega \rightarrow [0, \infty)$$ such that $$\int_{w\in \Omega} f(w)dw=1$$ and $$P(A) = \int_{w \in A} f(w)dw$$

확률 기타 부분

* 조건부 확률\(conditional probability\) $$P(A\vert B) \triangleq \dfrac{P(A \cap B)}{P(B)}$$
* 확률의 연쇄 법칙\(chain rule\): $$P(A \cap B) = P(A \vert B) P(B)$$
* 전체 확률의 법칙\(total probability law\): $$P(A) = P(A \cap B) + P(A \cap B^c) = P(A \vert B) P(B) + P(A \vert B^c) P(B^c)$$
* 베이즈 정리\(Bayes' rule\): $$P(B \vert A) = \dfrac{P(B \cap A)}{P(A)} = \dfrac{P(A \cap B)}{P(A)} = \dfrac{P(A \vert B)P(B)}{P(A)}$$
  * $$P(A \vert B)$$: likelihood
  * $$P(B \vert A)$$: posterior
  * $$P(B)$$: prior
* 독립 사건\(independent events\): $$P(A \cap B) = P(A) P(B)$$ 만 만족하면 independent한 것이다\($$\neq$$ disjoint, mutually exclusive\)
  * 예시:

    ![1.2.0.4](./figs/chapter-1/1.2.0.4-indpendentexam.png)

## Random Variable

* [확률 변수\(Random Variable\)](https://ko.wikipedia.org/wiki/확률_변수)는 측정가능한\(measureable\) [확률 공간\(Probability space\)](https://ko.wikipedia.org/wiki/확률_공간)-$$(\Omega, \mathcal{A}, P)$$과 [보렐 가측 공간\(Borel measureable space, 보통 실수들의 집합을 가르킴\)](https://ko.wikipedia.org/wiki/보렐_집합)-$$(\Bbb{R}, \mathcal{B})$$에서 정의되는 함수다.

  $$X: \Omega \rightarrow \Bbb{R} \text { such that } \forall B \in \mathcal{B}, X^{-1}(B) \in \mathcal{A}$$

  ![1.2.0.5](./figs/chapter-1/1.2.0.5-randomvariable.png)

* 여기서 랜덤\(random\)이란 확률 공간의 표본 공간\(sample space, $$\Omega$$\)에서 하나를 임의로 뽑는 과정을 가르킨다. `그림 1.2.0.5`와 같이 "숫자 4가 관측된다"라는 것을 풀어서 이야기하면 다음과 같다. 확률 공간의 표본 공간에서 임의로 뽑은 표본{4}를 확률 변수\($$X$$\)에 입력했을 때, 실수 공간\($$\Bbb{R}$$\)에 해당하는 숫자값 4를 부여하는 과정이다.   
* 이산 확률 변수\(\)

[확률 밀도 함수\(Probability density function\)](a1-edwithchoi.md) [상관분석\(Correlation analysis\)](a1-edwithchoi.md)

