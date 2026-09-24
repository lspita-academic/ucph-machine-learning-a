#set page(paper: "a4", margin: 2.5cm)
#set text(size: 12pt)
#set heading(numbering: "1.1")

// code style
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)
#set raw(tab-size: 4)
#show raw.where(block: true): block.with(
  fill: rgb("#F7F7F7"),
  stroke: 0.5pt + rgb("#EEEEEE"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
)

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

```py
counts_train = np.bincount(y_train)
freq_train = counts_train / counts_train.sum()
```

#figure(
  image("src/plots/classes_freqs.png", width: 70%),
  caption: [Classes frequencies bar chart],
) <fig:classes_freqs>

== Classification (30 points)

=== Logistic regression

The `LogisticRegression` class from `sklearn` is used to fit a model to the training dataset using logistic regression. For both the train and test datasets, the error is computed using the appropriate labels.

Regularization was not performed.

```py
lr = LogisticRegression(max_iter=150).fit(X_train, y_train)
lr_train_err = 1 - lr.score(X_train, y_train)
lr_test_err = 1 - lr.score(X_test, y_test)
```

Errors:
- Training set: $0.1496$
- Test set: $0.0993$

=== Random forest

The `RandomForestClassifier` class from `sklearn.ensemble` is used to fit a model to the training dataset using random forests. For both the train and test datasets, the error is computed using the appropriate labels. The out-of-bag error is also included.

Regularization was not performed.

```py
rf = RandomForestClassifier(n_estimators=ntrees, oob_score=True).fit(X_train, y_train)
rf_train_err = 1 - rf.score(X_train, y_train)
rf_test_err = 1 - rf.score(X_test, y_test)
rf_oob_err = 1 - rf.oob_score_
```

Errors:

#table(
  columns: 4,
  table.header[*Trees*][*Training err*][*Test err*][*OOB err*],
  [*50*], [$0.0004$], [$0.1132$], [$0.1519$],
  [*100*], [$2.9652 dot 10^(-5)$], [$0.1101$], [$0.1495$],
  [*200*], [$0$], [$0.1099$], [$0.1481$],
)

=== Nearest neighbor

The `KNeighborsClassifier` class from `sklearn.neighbors` is used to fit a model to the training dataset using k-nn classification.

First cross-validation is performed to determine the optimal value of `K`. Only odd values are used as candidates to avoid ties as much as possible. The range of values choosen is $[1, 200]$.

For each value of $K$, 5-fold cross validation is performed, then the candidate with the best average error across the folds is choosen as the optimal value of $K$.

```py
ks = np.arange(1, 202, 2)
cv = KFold(n_splits=5)
k_errs = []
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_errs = 1 - cross_val_score(knn, X_train, y_train, cv=cv)
    k_errs.append(np.mean(cv_errs))
K_idx = np.argmin(k_errs)
K = ks[K_idx]
K_err = k_errs[K_idx]
```

#figure(
  image("src/plots/cv_errs.png", width: 70%),
  caption: [Cross validation errors, with the best value for $K$ highlighted],
) <fig:cv_errs>

For both the train and test datasets, the error is computed using the appropriate labels.

Regularization was not performed.

```py
knn = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)
knn_train_err = 1 - knn.score(X_train, y_train)
knn_test_err = 1 - knn.score(X_test, y_test)
```

Errors:
- Training set: $0.1503$
- Test set: $0.0957$

= Overfitting (optional, 0 points)

// Optional reflection.
