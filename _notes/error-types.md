---
title: Testing the null hypothesis type I and type II errors
---

## Null hypothesis

We define the null hypothesis as the statement that we would like to prove is true. After the test is performed, we will be able to whether accept or reject the null hypothesis.

In this example, we want to predict the presence of spiders given the grain size. Hence, the null hypothesis consists in that the grain size is independent of the presence of spiders

![dataset](/assets/df_grain_spiders.png "Dataset view")

A logistic regression model is used to predict the presence of spiders given the grain size.

```python
lr_classifier = sklearn.linear_model.LogisticRegression(C=1e12, random_state=42)
lr_classifier.fit(df['grain_size'].values.reshape(-1, 1), df['spiders'])
print(lr_classifier.intercept_, lr_classifier.coef_)

[-1.64761964] [[5.12153717]]
```

Using the intercept and the coefficient, we can predict the presence of spiders given the grain size.

$$
probability of spider presence = \frac{1}{(1+e^{-1.6476+5.1215(grain \; size)}}
$$

![classifier](/assets/classifier%20output.png "Classifier output")

To test that, the likelihood ratio test (LRT) is performed.

$$
D = -2 \log{ \frac{L(H_0)}{L(H_1)} }
$$

### Null hypothesis model

A null model is a model that is used to test the null hypothesis. Here we fill all the grain sizes with zero, so that the presence of spiders is independent of the grain size.

```python
def null_likelyhood(y):
  clf = sklearn.linear_model.LogisticRegression(C=1e12)
  clf.fit(np.zeros_like(y).reshape(-1, 1), y)
  return clf

nullModel = null_likelyhood(df['spiders'])
```

Running the likelihood ratio test we get a p-value of 0.03. Hence, we reject the null hypothesis.


```python
def log_reg_lik_ratio_test(X, Y, clf0, clf1, df=1):
    if X.ndim == 1:
        X = X.values.reshape(-1, 1)
    y_prob0 = clf0.predict_proba(X)
    loss0 = sklearn.metrics.log_loss(Y, y_prob0, normalize=False)
    y_prob1 = clf1.predict_proba(X)
    loss1 = sklearn.metrics.log_loss(Y, y_prob1, normalize=False)
    D = 2 * (loss0 - loss1)
    return scipy.stats.distributions.chi2.sf(D, df=df) 

log_reg_lik_ratio_test(df['grain_size'], df['spiders'].astype(np.float64), nullModel, lr_classifier)    

0.033243766809119446
```

## Type I errors

The null hypothesis was true but it was rejected.

## Type II errors

The null hypothesis was false but it was accepted.