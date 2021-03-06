#' ---
#' title: "Supervised Learning - Regression Trees and Bagging"
#' author: "Jan-Philipp Kolb"
#' date: "`r format(Sys.time(), '%d %B, %Y')`"
#' fontsize: 10pt
#' output:
#'   beamer_presentation: 
#'     colortheme: dolphin
#'     fig_height: 3
#'     fig_width: 5
#'     fig_caption: no
#'     fonttheme: structuresmallcapsserif
#'     highlight: haddock
#'     theme: Dresden
#'   pdf_document: 
#'     keep_tex: yes
#'     toc: yes
#'   slidy_presentation: 
#'     css: mycss.css
#'     keep_md: yes
#' ---
#' 
## ----setupbaggingboosting, include=FALSE---------------------------------
knitr::opts_chunk$set(echo = T,message=F,warning=F,cache=T)
library(knitr)

#' 
#' 
#' ## [Tree-Based Models](https://www.statmethods.net/advstats/cart.html)
#' 
#' - Trees are good for interpretation because they are simple
#' 
#' - Tree based methods involve stratifying or segmenting the predictor space
#' into a number of simple regions. ([**Hastie and Tibshirani**](https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/trees.pdf))
#' 
#' ### But:
#' 
#' - These methods do not deliver the best results concerning prediction accuracy. 
#' 
#' <!--
#' https://lagunita.stanford.edu/courses/HumanitiesSciences/StatLearning/Winter2016/about
#' 
#' https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/trees.pdf
#' -->
#' 
#' <!--
#' 
#' https://en.wikipedia.org/wiki/Decision_tree_learning
#' 
#' https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf
#' 
#' https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
#' https://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/
#' -->
#' 
#' ## The Idea
#' 
#' - There are many methodologies for constructing regression trees but one of the oldest is the [**classification and regression tree**](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/) (CART) approach by Breiman et al. (1984). 
#' - Basic [**regression trees partition a data set into smaller subgroups**](https://towardsdatascience.com/the-complete-guide-to-decision-trees-28a4e3c7be14) and then fit a simple constant for each observation in the subgroup. 
#' - The partitioning is achieved by successive [**binary partitions**](https://en.wikipedia.org/wiki/Binary_space_partitioning) (aka [recursive partitioning](https://en.wikipedia.org/wiki/Recursive_partitioning)) based on the different predictors. 
#' - The constant to predict is based on the average response values for all observations that fall in that subgroup.
#' 
#' 
#' ## Explanation: [**decision tree**](https://en.wikipedia.org/wiki/Decision_tree)
#' 
#' <!--
#' https://elitedatascience.com/algorithm-selection
#' -->
#' 
#' Decision trees model data as a "tree" of hierarchical branches. They make branches until they reach "leaves" that represent predictions.
#' 
#' ![ [Decission tree on ElitedataScience](https://elitedatascience.com/algorithm-selection)](figure/Decision-Tree-Example.jpg)
#' 
#' ## Example decission trees
#' 
#' Due to their branching structure, decision trees can easily model nonlinear relationships.
#' 
#' ### Example
#' 
#' - For single family homes (larger lots) higher prices,
#' - and for apartments (smaller lots), also higher prices (because here it's a proxy for urban / rural).
#' 
#' 
#' This reversal of correlation is difficult for linear models to capture unless you explicitly add an interaction term
#' 
#' <!--
#' (i.e. you can anticipate it ahead of time).
#' -->
#'  -  Decision trees can capture this relationship naturally.
#' 
#' 
## ----echo=F--------------------------------------------------------------
library(rpart)

#' 
#' 
#' 
#' <!--
#' https://www.guru99.com/r-decision-trees.html
#' -->
#' 
#' ## Model foundations 
#' 
#' - This simple example can be generalized 
#' - We have a continuous response variable $Y$ and two inputs $X_1$ and $X_2$. 
#' - The recursive partitioning results in three regions ($R_1$,$R_2$,$R_3$) where the model predicts $Y$ with a constant $c_m$ for region $R_m$:
#' 
#' $$
#' \hat{f} (X) = \sum\limits_{m=1}^3c_mI(X_1,X_2)\in R_m
#' $$
#' <!--
#' An important question remains of how to grow a regression tree.
#' -->
#' 
#' ## How to grow a regression tree - deciding on splits
#' 
#' - It is important to realize the partitioning of variables are done in a top-down approach. 
#' - A partition performed earlier in the tree will not change based on later partitions. 
#' 
#' ### How are these partions made?
#' 
#' - The model begins with the entire data set, $S$, and searches every distinct value of every input variable to find the predictor and split value that partitions the data into two regions ($R_1$ and $R_2$) such that the overall sums of squares error are minimized:
#' 
#' $$
#' \text{minimize}\{SSE=\sum\limits_{i\in R_1}(y_i - c_1)^2 + \sum\limits_{i\in R_2} (y_i - c_2)^2 \}
#' $$
#' 
#' ## The best split 
#' 
#' - Having found the best split, we partition the data into the two resulting regions and repeat the splitting process on each of the two regions. 
#' - This process is continued until some stopping criterion is reached. 
#' - We typically get a very deep, complex tree that may produce good predictions on the training set, but is likely to [**overfit**](https://www.researchgate.net/post/What_is_over_fitting_in_decision_tree) the data, leading to poor performance on unseen data.
#' 
#' 
#' ## Regression Trees - preparation
#' 
#' - The following slides are based on the UC Business Analytics R Programming Guide on [**regression trees**](http://uc-r.github.io/regression_trees)
#' 
## ------------------------------------------------------------------------
library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging

#' 
## ----echo=F,eval=F-------------------------------------------------------
## install.packages("rpart.plot")

#' 
#' 
#' ## The Ames Housing data 
#' 
#' - Again we use the Ames dataset and split it in a test and training dataset
#' 
## ------------------------------------------------------------------------
set.seed(123)
ames_data <- AmesHousing::make_ames()
ames_split <- initial_split(ames_data,prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

#' 
#' 
#' ## Pruning
#' 
#' <!--
#' [**Boston housing data set**](http://lib.stat.cmu.edu/datasets/boston)
#' -->
#' 
#' - We create three decision trees based on three different samples of the data. 
#' - The first few partitions are fairly similar at the top of each tree; - they tend to differ closer to the terminal nodes. 
#' - These deeper nodes tend to overfit to specific attributes of the sample data; 
#' - Slightly different samples will result in highly variable estimate/ predicted values in the terminal nodes. 
#' - By [**pruning**](https://dzone.com/articles/decision-trees-and-pruning-in-r) these lower level decision nodes, we can introduce a little bit of bias in our model that help to stabilize predictions and will tend to generalize better to new, unseen data.
#' 
#' 
#' 
#' ## Three decision trees based on three samples.
#' 
#' ![](figure/decissiontree.PNG)
#' 
#' ## Cost complexity criterion
#' 
#' - There is often a balance to be achieved in the depth and complexity of the tree to optimize predictive performance on some unseen data. 
#' - To find this balance, we grow a very large tree as showed and then prune it back to find an optimal subtree. 
#' - We find this subtree by using a cost complexity parameter ($\alpha$) that penalizes our objective function for the number of terminal nodes of the tree (T).
#' 
#' $$
#' \text{minimize}\{SSE + \alpha|T|\}
#' $$
#' 
#' ## The complexity value $\alpha$  
#' 
#' - For a given value of $\alpha$, we find the smallest pruned tree that has the lowest penalized error. 
#' - $\rightarrow$ A close association to the lasso $L_1$ norm penalty. 
#' - Smaller penalties tend to produce more complex models, which result in larger trees. 
#' - Larger penalties result in much smaller trees. 
#' - As a tree grows larger, the reduction in the SSE must be greater than the cost complexity penalty. 
#' - We evaluate multiple models across a spectrum of $\alpha$ and use cross-validation to identify the optimal $\alpha$ and the optimal subtree.
#' 
#' ## Fit a regression tree using `rpart`
#' 
#' <!--
#' - We can fit a regression tree using `rpart` and then visualize it using `rpart.plot`.
#' -->
#' 
#' - The fitting process and the visual output of regression trees and classification trees are very similar. 
#' - Both use the formula method for expressing the model (similar to `lm`). 
#' - When fitting a regression tree, we need to set `method = "anova"`. 
#' - By default, `rpart` will make an intelligent guess based on the data type of the response column
#' - But it’s recommened to explictly set the method for reproducibility reasons (auto-guesser may change in future).
#' 
## ------------------------------------------------------------------------
m1 <- rpart(formula = Sale_Price ~ .,data = ames_train,
            method  = "anova")

#' 
#' ## the `m1` output. 
#' 
#' ![](figure/tree_m1.PNG)
#' 
#' 
## ------------------------------------------------------------------------
m1

#' 
#' 
#' 
#' ## Steps of the splits (m1) - explained
#' 
#' <!--
#' - Once we’ve fit our model we can look at the `m1` output. 
#' -->
#' 
#' 
#' <!--
#' - This just explains steps of the splits. 
#' -->
#' 
#' - E.g., we start with 2051 observations at the root node (very beginning) and the first variable we split on (that optimizes a reduction in SSE) is `Overall_Qual`. 
#' - We see that at the first node all observations with 
#' 
#' ```
#' Overall_Qual=Very_Poor,Poor,Fair,Below_Average,Average,
#' Above_Average,Good
#' ```
#' 
#' go to the 2nd branch. 
#' 
#' ## The 3rd branch
#' 
#' - The number of observations in this branch (1699), their average sales price (156147.10) and SSE (4.001092e+12) are listed. 
#' - In the 3rd branch we have 352 observations with
#' 
#' `Overall_Qual=Very_Good,Excellent,Very_Excellent` 
#' 
#' - their average sales prices is 304571.10 and the SEE in this region is 2.874510e+12. 
#' <!--
#' - `Overall_Qual` is the most important variable that has the largest reduction in SEE initially 
#' - With those homes on the upper end of the quality spectrum having almost double the average sales price.
#' 
#' -->
#' 
#' ### Visualization with `rpart.plot`
#' 
#' <!--
#' We can visualize our model with `rpart.plot`. 
#' -->
#' 
#' 
#' - `rpart.plot` has many plotting options
#' <!--
#' , which we’ll leave to the reader to explore. 
#' -->
#' - In the default print it will show the percentage of data that fall to that node and the average sales price for that branch. 
#' - This tree contains 11 internal nodes resulting in 12 terminal nodes. 
#' - This tree is partitioning on 11 variables to produce its model. 
#' <!--
#' fig.width=12, fig.height=8
#' -->
#' 
#' ## The package `rpart.plot`
#' 
## ----fig.height=6,fig.width=13-------------------------------------------
rpart.plot(m1)

#' 
#' 
#' ## Behind the scenes
#' 
#' There are 80 variables in `ames_train`. So what happened?
#' 
#' - `rpart` is automatically applying a range of cost complexity $\alpha$ values to prune the tree. 
#' - To compare the error for each $\alpha$ value, `rpart` performs a 10-fold cross validation so that the error associated with a $\alpha$ value is computed on the hold-out validation data. 
#' 
#' ## The `plotcp`
#' <!--
#' - Diminishing returns after 12 terminal nodes  (
#' - cross validation error - y-axis
#' - lower x-axis is cost complexity ($\alpha$) value, upper x-axis is the number of terminal nodes (tree size = $|T|$). 
#' 
#' -->
#' <!--
#' - The dashed line which goes through the point $|T|=9$. 
#' -->
#' 
#' - Lower x-axis - cost complexity - alpha
#' 
## ------------------------------------------------------------------------
plotcp(m1)

#' 
#' ## The 1-SE rule - how many terminal nodes
#' 
#' - Breiman et al. (1984) suggested to use the smallest tree within 1 standard deviation of the minimum cross validation error (aka the 1-SE rule). 
#' - Thus, we could use a tree with 9 terminal nodes and expect to get similar results within a small margin of error.
#' - To illustrate the point of selecting a tree with 12 terminal nodes (or 9 if you go by the 1-SE rule), we can force `rpart` to generate a full tree by using cp = 0 (no penalty results in a fully grown tree). 
#' 
#' ## Generate a full tree
#' 
#' - After 12 terminal nodes, we see diminishing returns in error reduction as the tree grows deeper. 
#' - Thus, we can signifcantly prune our tree and still achieve minimal expected error.
#' 
#' 
## ------------------------------------------------------------------------
m2 <- rpart(formula = Sale_Price ~ .,data=ames_train,
    method  = "anova",control = list(cp = 0, xval = 10))

#' 
#' - `control` - a list of options that control details of the rpart algorithm.
#' - `cp` - complexity parameter. Any split that does not decrease the overall lack of fit by a factor of cp is not attempted. For instance, with anova splitting, this means that the overall R-squared must increase by cp at each step (Pruning). 
#' - `xval`	number of cross-validations.
#' <!--
#' The main role of this parameter is to save computing time by pruning off splits that are obviously not worthwhile. Essentially,the user informs the program that any split which does not improve the fit by cp will likely be pruned off by cross-validation, and that hence the program need not pursue it.
#' -->
#' 
#' ## Plot the result
#' 
## ------------------------------------------------------------------------
plotcp(m2);abline(v = 12, lty = "dashed")

#' 
#' 
#' ## Automated tuning by default
#' 
#' - `rpart` is performing some automated tuning by default, with an optimal subtree of 11 splits, 12 terminal nodes, and a cross-validated error of 0.272 (note that this error is equivalent to the predicted residual error sum of squares statistic ([**PRESS**](https://en.wikipedia.org/wiki/PRESS_statistic))  but not the MSE). 
#' - We can perform additional tuning to try improve model performance.
#' 
#' ## The output `cptable`
#' 
## ------------------------------------------------------------------------
m1$cptable

#' 
#' ## Tuning
#' 
#' In addition to the cost complexity ($\alpha$) parameter, it is also common to tune:
#' 
#' ### `minsplit`: 
#' 
#' - The minimum number of data points required to attempt a split before it is forced to create a terminal node. The default is 20. Making this smaller allows for terminal nodes that may contain only a handful of observations to create the predicted value.
#' 
#' ### `maxdepth`: 
#' 
#' - The maximum number of internal nodes between the root node and the terminal nodes. The default is 30, which is quite liberal and allows for fairly large trees to be built.
#' 
#' ## Special control argument
#' 
#' - `rpart` uses a special control argument where we provide a list of hyperparameter values. 
#' - E.g., if we want a model with `minsplit = 10` and `maxdepth = 12`, we could execute the following:
#' 
## ------------------------------------------------------------------------
m3 <- rpart(formula = Sale_Price ~ .,data = ames_train,
    method  = "anova", control = list(minsplit = 10, 
                          maxdepth = 12, xval = 10)
)

#' 
#' ## The output `cptable` of model 3
#' 
## ------------------------------------------------------------------------
m3$cptable

#' 
#' ## Grid search
#' 
#' - We can avoid it to manually assess multiple models, by performing a grid search to automatically search across a range of differently tuned models to identify the optimal hyerparameter setting.
#' - To perform a grid search we first create our hyperparameter grid. 
#' <!--
#' - Here, we search a range of `minsplit` from 5-20 and vary `maxdepth` from 8-15 (since our original model found an optimal depth of 12). 
#' -->
#' 
#' <!--
#' ### Creating a hyper grid
#' -->
## ------------------------------------------------------------------------
hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)

#' 
#' - The result are 128 combinations - 128 different models.
#' 
## ------------------------------------------------------------------------
head(hyper_grid)
nrow(hyper_grid)

#' 
#' 
#' ## A loop to autimate modeling
#' 
#' <!--
#' - To automate the modeling we simply set up a for loop and 
#' -->
#' 
#' - We iterate through each `minsplit` and `maxdepth` combination. 
#' - We save each model into its own list item.
#' 
## ------------------------------------------------------------------------
models <- list()
for (i in 1:nrow(hyper_grid)) {
  # get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  # train a model and store in the list
  models[[i]] <- rpart(formula=Sale_Price~.,data=ames_train,
    method="anova",control=list(minsplit=minsplit,
                                maxdepth=maxdepth)
    )
}

#' 
#' ## A function to extract the minimum error
#' 
#' - We create functions to extract the minimum error associated with the optimal cost complexity $\alpha$ value for each model. 
#' 
## ------------------------------------------------------------------------
# function to get optimal cp
get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}

# function to get minimum error
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}

#' 
#' <!--
#' - After a little data wrangling to extract the optimal $\alpha$ value and its respective error, adding it back to our grid, and filter for the top 5 minimal error values we see that the optimal model makes a slight improvement over our earlier model (xerror of 0.242 versus 0.272).
#' 
#' -->
#' 
#' ## Apply the functions
#' 
## ------------------------------------------------------------------------
hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
    ) %>%
  arrange(error) %>%
  top_n(-5, wt = error)

#' 
#' ## Exercise
#' 
#' ### Apply the final optimal model
#' 
#' ### Predict on our test dataset
#' 
#' 
#' ## The final optimal model
#' 
#' <!--
#' - If we were satisfied with these results 
#' -->
#' ### Apply the final optimal model:
#' 
## ------------------------------------------------------------------------
optimal_tree <- rpart(formula = Sale_Price ~ .,
    data    = ames_train,method  = "anova",
    control = list(minsplit = 5, maxdepth = 13, cp = 0.0108982)
    )

#' 
#' ### Predict on our test dataset:
#' 
## ------------------------------------------------------------------------
pred <- predict(optimal_tree, newdata = ames_test)

#' 
#' 
#' 
#' - The final RMSE is 39145.39 which suggests that, on average, our predicted sales prices are about 39,145 Dollar off from the actual sales price.
#' 
## ------------------------------------------------------------------------
RMSE(pred = pred, obs = ames_test$Sale_Price)

#' 
#' 
#' ## [Exercise: `rpart` Kyphosis](https://www.r-exercises.com/2016/12/13/recursive-partitioning-and-regression-trees-exercises/)
#' 
#' ### Consider the Kyphosis data frame
#' 
#' <!--
#' (type `help("kyphosis")` for more details)
#' -->
#' 
#' 1) Which variables are in the `kyphosis` dataset
#' 2) Build a tree to classify Kyphosis from Age, Number and Start.
#' 
#' ### Consider the tree build above.
#' 
#' 3) Which variables are used to explain Kyphosis presence?
#' 4) How many observations contain the terminal nodes.
#' 
#' ### Consider the Kyphosis data frame.
#' 
#' 5) Build a tree using the first 60 observations of kyphosis.
#' 6) Predict the kyphosis presence for the other 21 observations.
#' 7) Which is the misclassification rate (prediction error)
#' 
#' ## Exercise: `rpart` `iris`
#' 
#' ### Consider the `iris` data frame
#' 
#' 1) Build a tree to classify Species from the other variables.
#' 2) Plot the trees, add nodes information.
#' 
#' ### Consider the tree build before
#' 
#' 3) Prune the the using median complexity parameter (cp) associated to the tree.
#' 4) Plot in the same window, the pruned and the original tree.
#' 5) In which terminal nodes is clasified each oobservations of `iris`?
#' 6) Which Specie has a flower of `Petal.Length` greater than 2.45 and `Petal.Width` less than 1.75.
#' 
#' <!--
#' ## What is bagging?
#' 
#' - Basic regression trees divide a data set into smaller groups and then fit a simple model (constant) for each subgroup. 
#' - But a single tree model tends to be highly unstable and a poor predictor. 
#' - Bootstrap aggregating (bagging) regression trees is quite powerful and effective. 
#' - This provides the fundamental basis of more complex tree-based models such as random forests and gradient boosting machines. 
#' -->
#' 
#' <!--
#' - This tutorial will get you started with regression trees and bagging.
#' 
#' ![]8figure/iris.png
#' -->
#' 
#' 
#' ## Advantages of regression trees
#' 
#' - They are very interpretable.
#' - Making predictions is fast (no complicated calculations, just looking up constants in the tree).
#' - It’s easy to understand what variables are important for the prediction. 
#' - The internal nodes (splits) are those variables that most largely reduced the SSE.
#' - If some data is missing, we might not be able to go all the way down the tree to a leaf, but we can still make a prediction by averaging all the leaves in the sub-tree.
#' - The model provides a non-linear response, so it can work when the true regression surface is not smooth. 
#' - If it is smooth, the piecewise-constant surface can approximate it arbitrarily closely (with enough leaves).
#' - There are fast, reliable algorithms to learn these trees.
#' 
#' ## Weaknesses of regression trees
#' 
#' - Single regression trees have high variance, resulting in unstable predictions (an alternative subsample of training data can significantly change the terminal nodes).
#' - Due to the high variance single regression trees have poor predictive accuracy.
#' 
#' <!--
#' 
#' ## [What are the advantages and disadvantages of decision trees?](https://elitedatascience.com/machine-learning-interview-questions-answers#supervised-learning)
#' 
#' Advantages: Decision trees are easy to interpret, nonparametric (which means they are robust to outliers), and there are relatively few parameters to tune.
#' 
#' Disadvantages: Decision trees are prone to be overfit. 
#' 
#' - This can be addressed by ensemble methods like random forests or boosted trees.
#' 
#' -->
#' 
#' ## [Ensembling](https://elitedatascience.com/overfitting-in-machine-learning)
#' 
#' Ensembles are machine learning methods for combining predictions from multiple separate models. 
#' 
#' <!--
#' There are a few different methods for ensembling, but the two most common are:
#' -->
#' 
#' ### Bagging 
#' 
#' attempts to reduce the chance of overfitting complex models.
#' 
#' 
#' - It trains a large number of "strong" learners in parallel.
#' -  A strong learner is a model that's relatively unconstrained.
#' -  Bagging then combines all the strong learners together in order to "smooth out" their predictions.
#' 
#' ### Boosting 
#' 
#' attempts to improve the predictive flexibility of simple models.
#' 
#' - It trains a large number of "weak" learners in sequence.
#' - A weak learner is a constrained model (limit for max depth of tree).
#' -    Each one in the sequence focuses on learning from the mistakes of the one before it.
#' - Boosting combines all the weak learners into a single strong learner.
#' 
#' ## Bagging and boosting
#' 
#' While bagging and boosting are both ensemble methods, they approach the problem from opposite directions.
#' 
#' Bagging uses complex base models and tries to "smooth out" their predictions, while boosting uses simple base models and tries to "boost" their aggregate complexity.
#' 
#' 
#' ## [Bagging](https://www.r-bloggers.com/improve-predictive-performance-in-r-with-bagging/)
#' 
#' - Single tree models suffer from high variance, they are highly unstable and poor predictors.
#' -  [**Pruning**](https://en.wikipedia.org/wiki/Decision_tree_pruning) helps, but there are alternative methods that exploite the variability of single trees in a way that can significantly improve performance. 
#' - Bootstrap aggregating (bagging) is one such approach (originally proposed by Breiman, 1996).
#' - Bagging is a method for combining predictions from different regression or classification models.
#' - The results of the models are then averaged - in the simplest case model predictions are included with the same weight.
#' - The weights could depend on the quality of the model prediction, i.e. "good" models are more important than "bad" models.
#' - Bagging leads to significantly improved predictions in the case of unstable models.
#' 
#' <!--
#' https://en.wikipedia.org/wiki/Bootstrap_aggregating
#' https://de.wikipedia.org/wiki/Bagging
#' http://topepo.github.io/caret/miscellaneous-model-functions.html#bagging-1
#' -->
#' 
#' <!--
#' ##
#' - Bagging combines and averages multiple models. 
#' - Averaging across multiple trees reduces the variability of any one tree and reduces overfitting, which improves predictive performance. 
#' -->
#' 
#' ## Bagging follows three simple steps:
#' 
#' - 1.) Create $m$ bootstrap samples from the training data. Bootstrapped samples allow us to create many slightly different data sets but with the same distribution as the overall training set.
#' 
#' - 2.) For each bootstrap sample train a single, unpruned regression tree.
#' 
#' - 3.) Average individual predictions from each tree to create an overall average predicted value.
#' 
#' ## The bagging process.
#' 
#' ![](figure/bagging3.png){ height=70% }
#' 
#' 
#' ## About bagging
#' 
#' - This process can be applied to any regression or classification model; 
#' - It provides the greatest improvement for models that have high variance. 
#' - More stable parametric models such as linear regression and multi-adaptive regression splines tend to experience less improvement in predictive performance.
#' - On average, a bootstrap sample will contain 63 per cent of the training data. 
#' - This leaves about 33 per cent ($\dfrac{1}{3}$) of the data out of the bootstrapped sample. We call this the out-of-bag (OOB) sample. 
#' - We can use the OOB observations to estimate the model’s accuracy, creating a natural cross-validation process.
#' 
#' ## Bagging with `ipred`
#' 
#' - Fitting a bagged tree model is quite simple. 
#' - Instead of using `rpart` we use `ipred::bagging`. 
#' - We use `coob = TRUE` to use the OOB sample to estimate the test error. 
#' 
#' ## Train bagged model
#' 
## ------------------------------------------------------------------------
set.seed(123)
(bagged_m1 <- bagging(formula = Sale_Price ~ .,
  data    = ames_train,coob= TRUE))

#' 
#' - We see that our initial estimate error is close to 3000 Dollar less than the test error we achieved with our single optimal tree (36543 vs. 39145)
#' 
#' ## Things to note typically
#' 
#' - The more trees the better - we are averaging over more high variance single trees. 
#' - We see a dramatic reduction in variance (and hence our error) and eventually the reduction in error will flatline 
#' <!--
#' - signaling an appropriate number of trees to create a stable model. 
#' -->
#' - You need less than 50 trees to stabilize the error.
#' 
#' ## Number of bootstrap samples
#' 
#' - By default bagging performs 25 bootstrap samples and trees but we may require more. 
#' <!--
#' - We can assess the error versus number of trees as below. 
#' -->
#' 
## ------------------------------------------------------------------------
# assess 10-50 bagged trees
ntree <- 10:50 
# create empty vector to store OOB RMSE values
rmse <- vector(mode = "numeric", length = length(ntree))
for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(123)   
  # perform bagged model
  model <- bagging(formula = Sale_Price ~ .,
  data=ames_train,coob= TRUE,nbagg=ntree[i]
)
  # get OOB error
  rmse[i] <- model$err   
}

#' 
#' 
#' ## Plot the result
#' 
#' - The error is stabilizing at about 25 trees - we will improve by bagging more trees.
#' 
#' 
## ------------------------------------------------------------------------
plot(ntree, rmse, type = 'l', lwd = 2)
abline(v = 25, col = "red", lty = "dashed")

#' 
#' 
#' 
#' ## Bagging with `caret`
#' 
#' - Bagging with `ipred` is simple but there are some additional benefits of bagging with `caret`.
#' 
#' 1.) Its easier to perform cross-validation. Although we can use the OOB error, performing cross validation will provide a more robust understanding of the true expected test error.
#' 
#' 2.) We can assess [**variable importance**](https://topepo.github.io/caret/variable-importance.html) across the bagged trees.
#' 
#' ## [Excursus: Variable importance (vi)](https://cran.r-project.org/web/packages/datarobot/vignettes/VariableImportance.html)
#' 
#' - vi measures help understand the results obtained from complex machine learning models 
#' - There is no general consensus on the “best” way to compute - or even define - the concept of variable importance. 
#' - See a list of many possible approaches to compute vi in the help file of the command `varImp` 
#' 
## ----eval=F--------------------------------------------------------------
## ?caret::varImp

#' 
#' - vi refers to how much a given model "uses" that variable to make accurate predictions. The more a model relies on a variable to make predictions, the more important it is for the model. 
#' 
#' <!--
#' https://stats.stackexchange.com/questions/332960/what-is-variable-importance
#' -->
#' 
#' <!--
#' 
#' 
#' https://topepo.github.io/caret/variable-importance.html
#' https://www.salford-systems.com/blog/dan-steinberg/what-is-the-variable-importance-measure
#' 
#' https://christophm.github.io/interpretable-ml-book/pdp.html
#' 
#' 
#' - We see that the cross-validated RMSE is 36,477 dollar. 
#' -->
#' 
#' 
#' ## A 10-fold cross-validated model.  
#' <!--
#' ## CV bagged model 
#' -->
## ------------------------------------------------------------------------
# Specify 10-fold cross validation
ctrl <- trainControl(method = "cv",  number = 10) 

bagged_cv <- train(Sale_Price ~ .,data = ames_train,
  method = "treebag",trControl = ctrl,importance = TRUE)

#' 
#' - `treebag`- means we use a bagging tree
#' 
#' ## Assess results
#' 
## ------------------------------------------------------------------------
bagged_cv

#' 
#' ## Assess results with a plot (top 20 variables)
#' 
#' <!--
#' - We also assess the top 20 variables from our model. 
#' -->
#' 
#' - Here, variable importance is measured by assessing the total amount SSE is decreased by splits over a given predictor, averaged over all $m$ trees. 
#' 
#' <!--
#' - The predictors with the largest average impact to SSE are considered most important. 
#' - The importance value is simply the relative mean decrease in SSE compared to the most important variable (provides a 0-100 scale).
#' -->
## ------------------------------------------------------------------------
plot(varImp(bagged_cv), 20) 

#' 
#' 
#' ## Extensions
#' 
#' - If we compare this to the test set out of sample we see that our cross-validated error estimate was very close. 
#' 
## ----predictbaggedcv-----------------------------------------------------
pred <- predict(bagged_cv, ames_test)
RMSE(pred, ames_test$Sale_Price)

#' 
#' - We have successfully reduced our error to about $35k; 
#' - Extensions of this bagging concept (random forests and GBMs) can significantly reduce this further.
#' 
#' <!--
#' ## [Classification Tree example](https://www.guru99.com/r-decision-trees.html)
#' 
#' The purpose of this dataset is to predict which people are more likely to survive after the collision with the iceberg. The dataset contains 13 variables and 1309 observations. The dataset is ordered by the variable X. 
#' 
#' 
## ------------------------------------------------------------------------
path <- 'https://raw.githubusercontent.com/thomaspernet/data_csv_r/master/data/titanic_csv.csv'
titanic <-read.csv(path)
shuffle_index <- sample(1:nrow(titanic))
kable(head(titanic))

#' -->
#' 
#' 
#' <!--
#' ## [**Decision trees used in data mining are of two main types:**](https://en.wikipedia.org/wiki/Decision_tree_learning)
#' 
#' - Classification tree
#' 
#' - Regression tree
#' -->
#' 
#' <!--
#' 
#' ## The data behind
#' 
## ------------------------------------------------------------------------
airq <- subset(airquality, !is.na(Ozone))
summary(airq$Temp)

#' 
#' 
#' 
#' ## Conditional inference tree
#' 
## ------------------------------------------------------------------------
library(party)

#' 
## ----eval=F--------------------------------------------------------------
## ?ctree

#' 
#' - performs recursively univariate split recursively
#' 
#' 
#' - [**Vignette**](https://cran.r-project.org/web/packages/party/vignettes/party.pdf) package `party`
#' 
#' 
#' ### [ctree example](https://datawookie.netlify.com/blog/2013/05/package-party-conditional-inference-trees/)
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("party")

#' 
#' ## A first model
#' 
## ----eval=F--------------------------------------------------------------
## library(party)

#' 
#' 
## ----eval=F--------------------------------------------------------------
## air.ct <- ctree(Ozone ~ ., data = airq, controls = ctree_control(maxsurrogate = 3))

#' 
#' 
#' ## The plot for `ctree`
#' 
## ----eval=F--------------------------------------------------------------
## plot(air.ct)

#' 
#' 
#' 
#' 
#' ## Recursive partitioning algorithms are special cases of a
#' simple two-stage algorithm
#' 
#' - First partition the observations by univariate splits in a recursive way and 
#' - second fit a constant model in each cell of the resulting partition.
#' 
#' 
#' ## [`ctree` - Regression](https://stats.stackexchange.com/questions/171301/interpreting-ctree-partykit-output-in-r)
#' 
## ------------------------------------------------------------------------
library(partykit)

#' 
## ----eval=F--------------------------------------------------------------
## ?ctree

#' 
## ------------------------------------------------------------------------
airq <- subset(airquality, !is.na(Ozone))
airct <- ctree(Ozone ~ ., data = airq)
plot(airct, type = "simple")

#' 
#' 
#' 
#' ## [Decision Trees](http://www.statmethods.net/advstats/cart.html)
#' 
#' [Regression tree vs. classification tree](http://www.statmethods.net/advstats/cart.html)
#' 
#' 
## ------------------------------------------------------------------------
library(rpart)

#' 
#' Grow a tree
#' 
## ------------------------------------------------------------------------
fit <- rpart(Kyphosis ~ Age + Number + Start,
   method="class", data=kyphosis)

printcp(fit) # display the results
plotcp(fit) # visualize cross-validation results
summary(fit) # detailed summary of splits

#' 
## ------------------------------------------------------------------------
# plot tree
plot(fit, uniform=TRUE,
   main="Classification Tree for Kyphosis")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

#' 
#' [Decision Trees and Random Forest](https://cran.r-project.org/doc/contrib/Zhao_R_and_data_mining.pdf)
#' 
#' -->
#' 
#' 
#' ## Resources and links
#' 
#' - Breimann (1984) - [**Classification and Regression Trees**](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)
#' 
#' - [**Vignette**](https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf) for package `partykit` 
#' 
#' - [**Conditional Inference Trees**](https://rpubs.com/awanindra01/ctree)
#' 
#' 
#' - [**Conditional inference trees vs traditional decision trees**](https://stats.stackexchange.com/questions/12140/conditional-inference-trees-vs-traditional-decision-trees)
#' 
#' - [Video on tree based methods](https://www.youtube.com/watch?v=6ENTbK3yQUQ)
#' 
#' - [An example of practical machine learning using R](https://rstudio-pubs-static.s3.amazonaws.com/64455_df98186f15a64e0ba37177de8b4191fa.html)
#' 
#' <!--
#' https://www.researchgate.net/figure/A-simple-example-of-visualizing-gradient-boosting_fig5_326379229
#' -->
#' 
#' 
#' 
#' 
