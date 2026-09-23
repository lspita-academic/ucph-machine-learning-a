#set page(paper: "a4", margin: 2.5cm)
#set text(size: 12pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Machine Learning A (2026--2027) \
    Home Assignment 3
  ]

  #text(fill: red)[Ludovico Maria Spitaleri, DMH249]
]

// Please leave the table of contents as is, for the ease of navigation for TAs.
#outline(indent: auto)
#pagebreak()

= Optimal Classification (16 points)

== Question 1 <q1.1>

Let us map the original label space $Y in {0, 1}$ to a new space $Y' in {-1, 1}$ following the original probabilities table for $X$ and $Y$.

We define our decision function based on the original probabilities:
$
  delta'(x) = P(Y' = 1 | X = x) - P(Y' = -1 | X = x)
$

The classification rule in the mapped space is:
$
  h'(x) = op("sgn")(delta'(x))
$

Evaluating this for the given inputs:
$
  cases(
    delta'(0) = 0 - 1 = -1 => h'(0) = op("sgn")(-1) = -1 & space "if" x = 0,
    delta'(1) = 0.2 - 0.8 = -0.6 => h'(1) = op("sgn")(-0.6) = -1 & space "if" x = 1
  ) \ \
  => h'(x) = -1
$

Mapping back to the original space $Y$, the optimal classification rule in the original space is $h(x) = 0$.

The expected risk (0-1 loss) is the probability of misclassification:
$
  R & = P(h(X) != Y) \
    & = sum_(x in X) P(X = x) P(h(x) != Y | X = x) \
    & = sum_(x in X) P(X = x) P(Y = 0 | X = x) \
    & = 0.5 dot 0 + 0.5 dot 0.2 = 0.1
$

== Question 2

In the case of $X = 0$, the probabilistic classifier still always predicts the correct label $Y = 0$, since $P(Y = 0 | X = 0) = 1$.

On the other hand, when considering $X = 1$ the expected risk is given by the probability of a mismatch between the classifier and the true label:
$
  & P(h(x) != Y | X = 1) \
    & space = P(h(x) = 0 | X = 1)P(Y = 1 | X = 1) + P(h(x) = 1 | X = 1)P(Y = 0 | X = 1) \
    & space = 0.8 dot 0.2 + 0.2 dot 0.8 = 0.32
$

The total expected risk is therefore:
$
  R & = P(h(X) != Y) \
    & = sum_(x in X) P(X = x) P(h(x) != Y | X = x) \
    & = 0.5 dot 0 + 0.5 dot 0.32 = 0.16
$

== Question 3

No, it is not possible to find a better classifier than the best one previously found, that is the deterministic classifier defined in @q1.1[section]. This is because by mapping the label space to $Y' = {-1, 1}$ the risk of this classifier corresponds to the Bayes risk, known to be the minimum overll all possible measurable functions $h$.
$
  h^("Bayes")(x) = op("sgn")[EE_(p)(Y | X = x)]
$

= Logistic Regression (50 points)

== Cross-entropy error measure (16 points)

=== Rewriting negative logarithmic likelihood as cross-entropy

// Write your derivation here.

=== Cross-entropy and logistic regression

// Write your argument here.


== Logistic regression loss gradient (24 points)

=== Gradient for $-1$ and $1$

// Write your derivation here.

=== Gradient for $0$ and $1$

// Write your derivation here.

=== Influence of misclassified examples

// Write your argument here.


== Log-odds (10 points)

// Write your proof here.


= Sleep Well (34 points)

== Data understanding and preprocessing (4 points)

The library `pandas` is used to load the datasets from the csv files. The classes frequencies are then calculated directly using `numpy`.

```python
counts_train = np.bincount(y_train)
freq_train = counts_train / counts_train.sum()
```

#figure(
  image("src/plots/classes_freqs.png", width: 70%),
  caption: [Classes frequencies bar chart],
) <fig:classes_freq_bar_chart>

== Classification (30 points)

=== Logistic regression

The `LogisticRegression` class from `sklearn` is used to fit a model to the training dataset. For both the train and test datasets, the prediction of the model is compared to the original labels to calculate the average error.

Regularization was not performed.

```py
lr = LogisticRegression(max_iter=150).fit(X_train, y_train)
lr_train_err = np.mean(lr.predict(X_train) != y_train)
lr_test_err = np.mean(lr.predict(X_test) != y_test)
```
Errors:
- Training set: $0.1496$
- Test set: $0.0993$

=== Random forest

// Deliverables: description of software used; training and test errors;
// out-of-bag error; description of regularization and model selection process, if used.

=== Nearest neighbor

// Deliverables: description of software used; training and test errors;
// description of regularization and model selection process, if used.


= Overfitting (optional, 0 points)

// Optional reflection.
