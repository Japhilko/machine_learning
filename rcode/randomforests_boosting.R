# Random Forests and Boosting

# Bagging suffers from tree correlation, which reduces the overall performance of the model.
# Random forests are a modification of bagging that builds a large collection of de-correlated trees
# Similar to bagging, each tree is grown to a bootstrap resampled data set, 
# which makes them different and decorrelates them.

library(rsample) # data splitting
library(randomForest) # basic implementation
library(ranger) # a faster implementation of randomForest
library(caret)


## The Ames housing data

load("../data/ames_data.RData")
set.seed(123)
ames_split <- rsample::initial_split(ames_data,prop=.7)
ames_train <- rsample::training(ames_split)
ames_test <- rsample::testing(ames_split)

############

set.seed(123)
# default RF model
(m1 <- randomForest(formula = Sale_Price ~ .,data=ames_train))

plot(m1)

# ntreeTry - We want enough trees to stabalize the error but using too
# many trees is inefficient, esp. for large data sets.

# mtry - number of variables as candidates at each split.
# When mtry=p -> bagging.
# When mtry=1 the split variable is completely random

  # package ranger is faster
library(ranger)
ames_ranger <- ranger(formula=Sale_Price ~ .,
                      data = ames_train,num.trees = 500,
                      mtry = floor(length(features) / 3))

ames_ranger
head(ames_ranger$predictions)

## tuning with a hypergrid

hyper_grid <- expand.grid(
  mtry = seq(20, 30, by = 2),
  node_size = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE = 0
)

nrow(hyper_grid)

for(i in 1:nrow(hyper_grid)) {
  model <- ranger(formula= Sale_Price ~ .,data= ames_train,
                  num.trees = 500,mtry= hyper_grid$mtry[i],
                  min.node.size = hyper_grid$node_size[i],
                  sample.fraction = hyper_grid$sampe_size[i],
                  seed = 123)
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% dplyr::arrange(OOB_RMSE) %>% head(10)

  # Variable importance

varimp_ranger <- optimal_ranger$variable.importance

lattice::barchart(sort(varimp_ranger)[1:25],col="royalblue")

pred_randomForest <- predict(ames_randomForest, ames_test)
head(pred_randomForest)

########################################################
# Boosting

library(rsample) # data splitting
library(gbm) # basic implementation
library(xgboost) # a faster implementation of gbm
library(caret) # aggregator package - machine learning
library(pdp) # model visualization
library(ggplot2) # model visualization
library(lime) # model visualization

ames_data <- AmesHousing::make_ames()
set.seed(123)
ames_split <- initial_split(ames_data,prop=.7)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

# distribution - depends on the response (e.g. bernoulli for binomial)
# n.tress - number of trees to fit
# interaction depth - 1 is for additive model
                    # 2 allows for 2-way interactions
# cv.folds - number of cross validation folds
# shrinkage - learning rate - a smaller learning rate typically requires more trees. 

gbm.fit <- gbm(formula = Sale_Price ~ .,distribution="gaussian",
               data = ames_train,n.trees = 100,interaction.depth = 1,
               shrinkage = 0.001,cv.folds = 5)

# this means on average our model is about $29,133 off from the actual sales price
sqrt(min(gbm.fit$cv.error))


# make prediction
pred <- predict(gbm.fit, ames_test)

