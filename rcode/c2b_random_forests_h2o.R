#' ---
#' title: "Random Forests with h2o"
#' author: "Jan-Philipp Kolb"
#' date: "24 Mai 2019"
#' output: html_document
#' ---
#' 
## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

#' 
#' ## The Ames housing data
#' 
## ------------------------------------------------------------------------
set.seed(123)
ames_split <- rsample::initial_split(AmesHousing::make_ames(), 
                                     prop = .7)
ames_train <- rsample::training(ames_split)
ames_test  <- rsample::testing(ames_split)

#' 
#' 
#' 
#' ## Full grid search with H2O
#' 
## ------------------------------------------------------------------------
library(h2o)          # an extremely fast java-based platform

#' 
#' 
#' - If you ran the grid search code above you probably noticed the code took a while to run. 
#' - `ranger` is computationally efficient, but as the grid search space expands, the manual for loop process becomes less efficient. 
#' - `h2o` is a powerful and efficient java-based interface that provides parallel distributed algorithms. 
#' - `h2o` allows for different optimal search paths in our grid search. This allows us to be more efficient in tuning our models. Here, I demonstrate how to tune a random forest model with `h2o`. Lets go ahead and start up h2o:
#' 
#' <!--
#' (I turn off progress bars when creating reports/tutorials)
#' -->
#' 
#' 
#' 
## ------------------------------------------------------------------------
# start up h2o 
h2o.no_progress()
h2o.init(max_mem_size = "5g")

#' 
#' ## Random forests with `h2o`
#' 
#' - We can try a comprehensive (full cartesian) grid search, which means we will examine every combination of hyperparameter settings that we specify in `hyper_grid.h2o`. 
#' - We search across 96 models but since we perform a full cartesian search this process is not any faster. 
#' - Note that the best performing model has an OOB RMSE of 24504, which is lower than what we achieved previously. 
#' - This is because some of the default settings regarding minimum node size, tree depth, etc. are more “generous” than `ranger` and `randomForest` 
#' - E.g. `h2o` has a default minimum node size of one whereas `ranger` and `randomForest` default settings are 5.
#' 
#' 
#' ## Preparation for `h2o`
#' 
## ------------------------------------------------------------------------
# create feature names
y <- "Sale_Price"
x <- setdiff(names(ames_train), y)
# turn training set into h2o object
train.h2o <- as.h2o(ames_train)
# hyperparameter grid
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 100),
  mtries      = seq(20, 30, by = 2),
  sample_rate = c(.55, .632, .70, .80)
)

#' 
#' ##
#' 
#' <!--
#' Achtung: folgendes dauert sehr lange
#' -->
#' 
## ----eval=F--------------------------------------------------------------
## # build grid search
## grid <- h2o.grid(
##   algorithm = "randomForest",
##   grid_id = "rf_grid",
##   x = x,
##   y = y,
##   training_frame = train.h2o,
##   hyper_params = hyper_grid.h2o,
##   search_criteria = list(strategy = "Cartesian")
##   )

#' 
## ----eval=F,echo=F-------------------------------------------------------
## save(grid,file="../data/ml_rf_h2o.grid.RData")

#' 
## ----echo=F--------------------------------------------------------------
load("../data/ml_rf_h2o.grid.RData")

#' 
#' 
## ------------------------------------------------------------------------
# collect the results and sort by our model performance 
# metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "rf_grid", 
  sort_by = "mse", 
  decreasing = FALSE
  )

#' 
#' ##
#' 
## ----eval=F,echo=F-------------------------------------------------------
## save(grid_perf,file = "../data/ml_rf_grid_perf.data")

#' 
## ----echo=F,eval=T-------------------------------------------------------
load("../data/ml_rf_grid_perf.data")

#' 
#' 
## ----eval=T--------------------------------------------------------------
print(grid_perf)

#' 
#' 
#' ## Combinatorial explosion
#' 
#' - Because of the [**combinatorial explosion**](https://en.wikipedia.org/wiki/Combinatorial_explosion), each additional hyperparameter added has a huge effect on the time. 
#' - `h2o` provides an additional grid search path called “RandomDiscrete”, which will jump from one random combination to another and stop once a certain level of improvement has been made, certain amount of time has been exceeded, or a certain amount of models have been ran (or a combination of these have been met). 
#' - A random discrete search path will likely not find the optimal model, but it does a good job of finding a very good model.
#' 
#' - E.g., the following code searches 2,025 hyperparameter combinations. 
#' - Our random grid search will stop if none of the last 10 models provides a 0.5% improvement in MSE. 
#' - If we continue to find improvements then I cut the grid search off after 600 seconds (30 minutes). 
#' - Our grid search assessed 190 models and the best model (max_depth = 30, min_rows = 1, mtries = 25, nbins = 30, ntrees = 200, sample_rate = .8) achived an RMSE of 24686 
#' ).
#' 
#' ##
#' 
## ------------------------------------------------------------------------
# hyperparameter grid
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(15, 35, by = 10),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(1, 5, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .75)
)

#' 
## ------------------------------------------------------------------------
# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 30*60
  )

#' 
#' 
#' ##
#' 
#' <!--
#' Folgendes dauert wieder lange
#' -->
#' 
## ------------------------------------------------------------------------
# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = x, 
  y = y, 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = search_criteria
  )

#' 
## ------------------------------------------------------------------------
# collect the results and sort by our model performance 
# metric of choice
grid_perf2 <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
  )

#' 
## ----eval=F,echo=F-------------------------------------------------------
## save(random_grid,grid_perf2,file="../data/ml_rf_random_grid.RData")

#' 
## ----echo=F,eval=T-------------------------------------------------------
load("../data/ml_rf_random_grid.RData")

#' 
#' 
#' ## 
#' 
## ----eval=T--------------------------------------------------------------
print(grid_perf2)

#' 
#' ## Hold-out test
#' 
#' - Once we’ve identifed the best model we can get that model and apply it to our hold-out test set to compute our final test error. 
#' 
## ------------------------------------------------------------------------
# Grab the model_id for the top model, 
# chosen by validation error
best_model_id <- grid_perf2@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

#' 
#' <!--
#' Folgendes funktioniert nicht - man braucht ein größeres Cluster
#' -->
#' 
## ----eval=F--------------------------------------------------------------
## # Now let’s evaluate the model performance on a test set
## ames_test.h2o <- as.h2o(ames_test)
## best_model_perf <- h2o.performance(model = best_model,
##                                    newdata = ames_test.h2o)
## 
## # RMSE of best model
## h2o.mse(best_model_perf) %>% sqrt()

#' 
#' 
#' - We have reduced our RMSE to near 23,000, which is a 10K reduction compared to elastic nets and bagging.
#' 
#' 
#' ## Links
#' 
#' - [Download h2o](http://h2o.ai/download/)
#' 
