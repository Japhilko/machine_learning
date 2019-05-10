---
title: "Lasso Regression"
author: "Jan-Philipp Kolb"
date: "03 Mai, 2019"
output: 
  slidy_presentation: 
    keep_md: yes
---




## [Regularization](https://elitedatascience.com/algorithm-selection)

### elitedatascience.com definition

Regularization is a technique used to prevent overfitting by artificially penalizing model coefficients.

- It can discourage large coefficients (by dampening them).
-  It can also remove features entirely (by setting their coefficients to 0).
-  The "strength" of the penalty is tunable. (More on this tomorrow...)

### Wikipedia definition of [Regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) 


Regularization is the process of adding information in order to solve an ill-posed problem or to prevent overfitting. 


## Strenghts and weaknesses of [regularization](https://elitedatascience.com/machine-learning-algorithms)

Regularization is a technique for penalizing large coefficients in order to avoid overfitting, and the strength of the penalty should be tuned.

### Strengths: 

Linear regression is straightforward to understand and explain, and can be regularized to avoid overfitting. In addition, linear models can be updated easily with new data

### Weaknesses: 

Linear regression performs poorly when there are non-linear relationships. They are not naturally flexible enough to capture more complex patterns, and adding the right interaction terms or polynomials can be tricky and time-consuming.


## [Regularized Regression Algos](https://elitedatascience.com/algorithm-selection)

![](figure/reg_3algos.PNG)

## [Lasso regression](https://elitedatascience.com/algorithm-selection)

LASSO, stands for least absolute shrinkage and selection operator 


-  Lasso regression penalizes the absolute size of coefficients.
-   Practically, this leads to coefficients that can be exactly 0.
-   Thus, Lasso offers automatic feature selection because it can completely remove some features.
-   Remember, the "strength" of the penalty should be tuned.
-  A stronger penalty leads to more coefficients pushed to zero.


## [Ridge regression](https://elitedatascience.com/algorithm-selection)


-    Ridge regression penalizes the squared size of coefficients.
-    Practically, this leads to smaller coefficients, but it doesn't force them to 0.
-    In other words, Ridge offers feature shrinkage.
-    Again, the "strength" of the penalty should be tuned.
-    A stronger penalty leads to coefficients pushed closer to zero.


## [Elastic net](https://elitedatascience.com/algorithm-selection)



- Elastic-Net is a compromise between Lasso and Ridge.

-Elastic-Net penalizes a mix of both absolute and squared size.
    - The ratio of the two penalty types should be tuned.
    - The overall strength should also be tuned.

- There’s no "best" type of penalty. It depends on the dataset and the problem. 
- We recommend trying different algorithms that use a range of penalty strengths as part of the tuning process



## [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) regression overview

- lasso is a regression analysis method that performs variable selection and regularization (reduce overfitting)
- We want to enhance prediction accuracy and interpretability of the statistical model.

<!--
https://eight2late.wordpress.com/2017/07/11/a-gentle-introduction-to-logistic-regression-and-lasso-regularisation-using-r/
-->

- We could remove less important variables, after checking that they are not important.
- We can do that manually by examining p-values of coefficients and discarding those variables whose coefficients are not significant.
- This can become tedious for classification problems with many independent variables


## History of lasso

- Originally introduced in geophysics literature in 1986
- Independently rediscovered and popularized in 1996 by Robert Tibshirani, who coined the term and provided further insights into the observed performance.



Lasso was originally formulated for least squares models and this simple case reveals a substantial amount about the behavior of the estimator, including its relationship to ridge regression and best subset selection and the connections between lasso coefficient estimates and so-called soft thresholding. It also reveals that (like standard linear regression) the coefficient estimates need not be unique if covariates are collinear.

## Lasso for other models than least squares

Though originally defined for least squares, lasso regularization is easily extended to a wide variety of statistical models including generalized linear models, generalized estimating equations, proportional hazards models, and M-estimators, in a straightforward fashion.

- Lasso’s ability to perform subset selection relies on the form of the constraint and has a variety of interpretations including in terms of geometry, Bayesian statistics, and convex analysis.

The LASSO is closely related to basis pursuit denoising.




## What is [lasso regression](http://www.statisticshowto.com/lasso-regression/)

- Lasso regression uses shrinkage
- data values are shrunk towards a central point

- [Ridge and lasso regularization work by adding a penalty term to the log likelihood function.](https://eight2late.wordpress.com/2017/07/11/a-gentle-introduction-to-logistic-regression-and-lasso-regularisation-using-r/)

- A tuning parameter, $\lambda$ controls the strength of the L1 penalty.

$$
\sum\limits_{i=1}^n \big( y_i -\beta_0 - \sum\limits_{j=1}^p \beta_jx_{ij} \big)^2 + \lambda \sum\limits_{j=1}^p |\beta_j| = RSS + \lambda\sum\limits_{j=1}^p |\beta_j|.
$$
<!--
wir haben einen penalty term, der hoch ist, wenn die Parameterschätzwerte hoch sind.

Youtube Video zu Lasso
https://www.youtube.com/watch?v=A5I1G1MfUmA
-->


## [The L1 norm explained](https://stats.stackexchange.com/questions/347257/geometrical-interpretation-of-l1-regression)

![](figure/BBRXC.png)

## [Ridge Regression and the Lasso](https://www.r-bloggers.com/ridge-regression-and-the-lasso/)

### Import of build in data


```r
swiss <- datasets::swiss
```

### First impression of the data


```r
library(DT)
DT::datatable(swiss)
```

<!--html_preserve--><div id="htmlwidget-abe9112e0c33ce5ef209" style="width:100%;height:auto;" class="datatables html-widget"></div>
<script type="application/json" data-for="htmlwidget-abe9112e0c33ce5ef209">{"x":{"filter":"none","data":[["Courtelary","Delemont","Franches-Mnt","Moutier","Neuveville","Porrentruy","Broye","Glane","Gruyere","Sarine","Veveyse","Aigle","Aubonne","Avenches","Cossonay","Echallens","Grandson","Lausanne","La Vallee","Lavaux","Morges","Moudon","Nyone","Orbe","Oron","Payerne","Paysd'enhaut","Rolle","Vevey","Yverdon","Conthey","Entremont","Herens","Martigwy","Monthey","St Maurice","Sierre","Sion","Boudry","La Chauxdfnd","Le Locle","Neuchatel","Val de Ruz","ValdeTravers","V. De Geneve","Rive Droite","Rive Gauche"],[80.2,83.1,92.5,85.8,76.9,76.1,83.8,92.4,82.4,82.9,87.1,64.1,66.9,68.9,61.7,68.3,71.7,55.7,54.3,65.1,65.5,65,56.6,57.4,72.5,74.2,72,60.5,58.3,65.4,75.5,69.3,77.3,70.5,79.4,65,92.2,79.3,70.4,65.7,72.7,64.4,77.6,67.6,35,44.7,42.8],[17,45.1,39.7,36.5,43.5,35.3,70.2,67.8,53.3,45.2,64.5,62,67.5,60.7,69.3,72.6,34,19.4,15.2,73,59.8,55.1,50.9,54.1,71.2,58.1,63.5,60.8,26.8,49.5,85.9,84.9,89.7,78.2,64.9,75.9,84.6,63.1,38.4,7.7,16.7,17.6,37.6,18.7,1.2,46.6,27.7],[15,6,5,12,17,9,16,14,12,16,14,21,14,19,22,18,17,26,31,19,22,14,22,20,12,14,6,16,25,15,3,7,5,12,7,9,3,13,26,29,22,35,15,25,37,16,22],[12,9,5,7,15,7,7,8,7,13,6,12,7,12,5,2,8,28,20,9,10,3,12,6,1,8,3,10,19,8,2,6,2,6,3,9,3,13,12,11,13,32,7,7,53,29,29],[9.96,84.84,93.4,33.77,5.16,90.57,92.85,97.16,97.67,91.38,98.61,8.52,2.27,4.43,2.82,24.2,3.3,12.11,2.15,2.84,5.23,4.52,15.14,4.2,2.4,5.23,2.56,7.72,18.46,6.1,99.71,99.68,100,98.96,98.22,99.06,99.46,96.83,5.62,13.79,11.22,16.92,4.97,8.65,42.34,50.43,58.33],[22.2,22.2,20.2,20.3,20.6,26.6,23.6,24.9,21,24.4,24.5,16.5,19.1,22.7,18.7,21.2,20,20.2,10.8,20,18,22.4,16.7,15.3,21,23.8,18,16.3,20.9,22.5,15.1,19.8,18.3,19.4,20.2,17.8,16.3,18.1,20.3,20.5,18.9,23,20,19.5,18,18.2,19.3]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th> <\/th>\n      <th>Fertility<\/th>\n      <th>Agriculture<\/th>\n      <th>Examination<\/th>\n      <th>Education<\/th>\n      <th>Catholic<\/th>\n      <th>Infant.Mortality<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"columnDefs":[{"className":"dt-right","targets":[1,2,3,4,5,6]},{"orderable":false,"targets":0}],"order":[],"autoWidth":false,"orderClasses":false}},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->


### Preparing the model


```r
x <- model.matrix(Fertility~., swiss)[,-1]
y <- swiss$Fertility
lambda <- 10^seq(10, -2, length = 100)
```


## Test and train dataset


```r
library(glmnet)
set.seed(489)
train = sample(1:nrow(x), nrow(x)/2)
test = (-train)
ytest = y[test]
```


## A first ols model


```r
#OLS
swisslm <- lm(Fertility~., data = swiss)
coef(swisslm)
```

```
##      (Intercept)      Agriculture      Examination        Education 
##       66.9151817       -0.1721140       -0.2580082       -0.8709401 
##         Catholic Infant.Mortality 
##        0.1041153        1.0770481
```

## A ridge model


```r
#ridge
ridge.mod <- glmnet(x, y, alpha = 0, lambda = lambda)
predict(ridge.mod, s = 0, type = 'coefficients')[1:6,]
```

```
##      (Intercept)      Agriculture      Examination        Education 
##       66.8911177       -0.1714307       -0.2603091       -0.8681376 
##         Catholic Infant.Mortality 
##        0.1037196        1.0776950
```


## Lasso regression with package `glmnet`


```r
install.packages("glmnet")
```


```r
library(glmnet)
```


```r
x=matrix(rnorm(100*20),100,20)
g2=sample(1:2,100,replace=TRUE)
fit2=glmnet(x,g2,family="binomial")
```


```r
caret::varImp(fit2,lambda=0.0007567)
```

```
##         Overall
## V1  0.150397416
## V2  0.208434920
## V3  0.633739581
## V4  0.546691236
## V5  0.040870939
## V6  0.131795416
## V7  0.039071452
## V8  0.131155125
## V9  0.200379120
## V10 0.192028538
## V11 0.596070142
## V12 0.299850144
## V13 0.376385061
## V14 0.327054714
## V15 0.135271420
## V16 0.006725415
## V17 0.255749840
## V18 0.136566489
## V19 0.331609952
## V20 0.229067861
```


## 

- LASSO is a feature selection method.
<!--
https://eight2late.wordpress.com/2017/07/11/a-gentle-introduction-to-logistic-regression-and-lasso-regularisation-using-r/
-->
- LASSO regression has inbuilt penalization functions to reduce overfitting.
<!--
https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/
-->


## 

- The logarithmic function is used for the link between probability and logits

- The Logit function is used to [linearize sigmoid curves](https://de.wikipedia.org/wiki/Logit).

<!--
Die Logit-Funktion wird zur Linearisierung von sigmoiden Kurven verwendet.
-->

## The package `caret`

- Classification and Regression Training


```r
install.packages("caret")
```


```r
library("caret")
```

- [**Vignette `caret` package **](https://cran.r-project.org/web/packages/caret/vignettes/caret.html) - 

## 


```r
?caret::train
```




```r
logit<-train(,data = gp.train.c,
                        method = 'glm',
                        family = 'binomial',
                        trControl = ctrl0)")
```


## Further packages 


```r
# https://cran.rstudio.com/web/packages/biglasso/biglasso.pdf
install.packages("biglasso")
```



## Links


[A comprehensive beginners guide for Linear, Ridge and Lasso Regression](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)

- Course for statistical learning - [Youtube - Videos](https://www.r-bloggers.com/in-depth-introduction-to-machine-learning-in-15-hours-of-expert-videos/)

- [pcLasso: a new method for sparse regression](https://www.r-bloggers.com/pclasso-a-new-method-for-sparse-regression/)

- [Youtube - lasso regression - clearly explained](https://www.youtube.com/watch?v=NGf0voTMlcs) 

- [Glmnet Vignette](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html)

- [Regularization Methods in R](https://www.geo.fu-berlin.de/en/v/soga/Geodata-analysis/multiple-regression/Regularization-Methods/Regularization-Methods-in-R/index.html)

- [A gentle introduction to logistic regression and lasso regularisation using R](https://eight2late.wordpress.com/2017/07/11/a-gentle-introduction-to-logistic-regression-and-lasso-regularisation-using-r/)

- [Penalized Regression in R](https://machinelearningmastery.com/penalized-regression-in-r/)

- [Penalized Logistic Regression Essentials in R](http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/)

- [All you need to know about Regularization](https://towardsdatascience.com/all-you-need-to-know-about-regularization-b04fc4300369)
