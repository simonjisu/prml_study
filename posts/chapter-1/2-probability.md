패턴인식에서 중요한 개념 중 하나는 **불확실성(uncertainty)** 이다. **확률 이론(Probability Theory)** 은 불확실성을 정확하고 양적인 방식으로 측정할 수 있는 일관된 프레임워크를 제공해준다. 또한 결정 이론(Decision Theory) 와 결합하면 현재 가진 정보내에서 최적의 예측을 내릴수 있도록 도와준다.

# Set Theory 부터 Probability Theory 까지 빠르게 훑기

## Set Theory

> 최성준님의 [베이지안 딥러닝](https://www.edwith.org/bayesiandeeplearning/) 자료를 많이 참고했습니다.

집합론(set theory)은 추상적 대상들의 모임인 집합을 연구하는 수학 이론이다. 기본적인 개념은 위키링크를 달아 두었다.

* [집합(set)](https://ko.wikipedia.org/wiki/집합): 특정 조건에 맞는 원소들의 모임
* [원소(element)](https://ko.wikipedia.org/wiki/원소_\(수학\) ): 집합을 이루는 개체, 원소 $$a$$가 집합 $$A$$에 속할 경우 $$a \in A$$라고 표기한다.
* [부분 집합(subset)](https://ko.wikipedia.org/wiki/부분집합): 집합 A의 모든 원소가 다른 집합 B에도 속하는 관계일 경우, A는 B의 "부분 집합"이라고 한다.
* [전체집합(universal set)](https://ko.wikipedia.org/wiki/전체집합): 모든 대상(자기 자신도 포함)을 원소로 포함하는 집합
* [집합의 연산(set operations)](https://en.wikipedia.org/wiki/Set_\(mathematics\)#Basic_operations) 
    * [합집합(Unions)](https://ko.wikipedia.org/wiki/합집합)
    * [교집합(Intersections)](https://ko.wikipedia.org/wiki/교집합)
    * [여집합(Complements)](https://ko.wikipedia.org/wiki/여집합)
    * [곱집합(product set, Cartesian product)](https://ko.wikipedia.org/wiki/곱집합): 각 집합의 원소를 각 선분으로 하는 튜플(tuple)들의 집합 
        
        $$ A \times B = \{ (a, b): \mathtt{a} \in A, \mathtt{b} \in B\}$$
        
        * 예시: $$A = \{ 1, 2 \}, B = \{ 3, 4, 5 \} \rightarrow A \times B = \{ (1,3), (1,4), (1,5), (2,3), (2,4), (2,5) \}$$
* [서로소 집합(disjoint set)](https://ko.wikipedia.org/wiki/서로소_집합): 공통 원소가 없는 두 집합, $$A \cap B = \emptyset $$
* [집합의 분할(partition of a set)](https://ko.wikipedia.org/wiki/집합의_분할): 집합의 원소들을 비공 부분 집합들에게 나눠주어, 모든 원소가 각자 정확히 하나의 부분 집합에 속하게끔 하는 것
    * 예시: $$A = \{ 1, 2, 3, 4 \} \rightarrow \text{partition of set A} = \{ \{1, 2\}, \{3\}, \{4\} \}$$
* [멱잡합(power set of set A, $$\coloneqq 2^A$$)](https://ko.wikipedia.org/wiki/멱집합): 주어진 집합의 모든 부분 집합들로 구성된 집합(the set of all the subsets)
    * 예시: $$A = \{ 1, 2, 3 \} \rightarrow \text{power set of 2}^A = \{ \emptyset, \{1\},\{2\},\{3\},\{1,2\},\{2,3\},\{1,3\},\{1,2,3\} \} $$
* [집합의 크기(Cardinality)](https://ko.wikipedia.org/wiki/집합의_크기): 집합의 "원소 개수"에 대한 척도, $$\vert A \vert $$로 표기 한다. 집합의 크기를 표현하는 용어로 finite, infinite, countable, uncountable, denumerable(countably infinite)가 있다.
    * [가산 집합(countable set)](https://ko.wikipedia.org/wiki/가산_집합): 관심있는 집합과 자연수의 집합으로 [일대일 함수](https://ko.wikipedia.org/wiki/단사_함수)(one-to-one function)관계가 존재하면, 그 집합은 가산 집합이다. 특히, 자연수, 정수, 유리수와 같이 셀수 있는 무한 집합의 경우, 가산 무한(countable infinite)이나 가부번 집합(denumerable set)이라고 한다.
    * 비가산 집합(uncountable set): 가산 집합이 아닌 집합, 실수는 비가산 집합

## Function

![1.2.1](/figs/chapter-2/1.2.1-function.png)

<img src="/figs/chapter-2/1.2.1-function.png" desc="1.2.1" width="75%" >

* [함수/사상(function/mapping)](https://ko.wikipedia.org/wiki/함수): 첫 번째 집합의 임의의 한 원소를 두 번째 집합의 오직 한 원소에 대응시키는 이항 관계이다. 입력이 되는 집합 $$U$$를 정의역(domain), 출력으로 대응되는 집합 $$V$$를 공역(codomain)이라고 한다. 
    
    $$f: \underset{domain}{U} \rightarrow \underset{codomain}{V}$$

* [상(image)](https://ko.wikipedia.org/wiki/상_\(수학\)): domain의 원소(혹은 부분 집합)가 대응하는 codomain의 원소(혹은 집합) 
    
    $$f(x) \in V, x \in U \quad \text{or} \quad f(A) = \{ f(x) \vert x \in A \} \sube V, A \sube U $$

    반대로 codomain의 원소에 대응하는 domain의 원소를 역상(inverse image)이라고 한다(원소의 역상은 부분 집합이라는 것을 주의).

    $$f^{-1}(y) = \{ x \in U \vert f(x) \in V \} \sube V \quad \text{or} \quad f^{-1}(B) = \{ x \vert f(x) \in B \} \sube U, B \sube V$$
* [치역(range)](https://ko.wikipedia.org/wiki/치역): 함수의 모든 출력값의 집합, 치역은 공역(codomain)의 부분 집합이다.

![1.2.2](/figs/chapter-2/1.2.2-invertible.png)

* [일대일 함수/단사 함수(one-to-one/injective)](https://ko.wikipedia.org/wiki/단사_함수): domain의 서로 다른 원소를 codimain의 서로 다른 원소로 대응시키는 함수
* [위로의 함수/전사 함수(onto/surjective)](https://ko.wikipedia.org/wiki/전사_함수): domain과 range가 일치하는 함수
* one-to-one 조건과 onto 조건을 모두 만족하면 가역 함수(invertible function)라고 한다.

## Probability

