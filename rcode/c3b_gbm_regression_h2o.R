#' ---
#' title: "Gradient boosting with h2o"
#' author: "Jan-Philipp Kolb"
#' date: "24 Mai 2019"
#' output: ioslides_presentation
#' ---
#' 
## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = FALSE)

#' 
#' 
#' ## h2o
#' 
## ------------------------------------------------------------------------
library(h2o)          # a java-based platform

#' 
#' 
#' The h2o R package is a powerful and efficient java-based interface that allows for local and cluster-based deployment. It comes with a fairly comprehensive online resource that includes methodology and code documentation along with tutorials.
#' 
#' ## Features include:
#' 
#' - Distributed and parallelized computation on either a single node or a multi-node cluster.
#' - Automatic early stopping based on convergence of user-specified metrics to user-specified relative tolerance.
#' - Stochastic GBM with column and row sampling (per split and per tree) for better generalization.
#' - Support for exponential families (Poisson, Gamma, Tweedie) and loss functions in addition to binomial (Bernoulli), Gaussian and multinomial distributions, such as  Quantile regression (including Laplace).
#' - Grid search for hyperparameter optimization and model selection.
#' - Data-distributed, which means the entire dataset does not need to fit into memory on a single node, hence scales to any size training set.
#' - Uses histogram approximations of continuous variables for speedup.
#' - Uses dynamic binning - bin limits are reset at each tree level based on the split bins’ min and max values discovered during the last pass.
#' - Uses squared error to determine optimal splits.
#' <!--
#' - Distributed implementation details outlined in a blog post by Cliff Click.
#' -->
#' - Unlimited factor levels.
#' - Multiclass trees (one for each class) built in parallel with each other.
#' - Apache 2.0 Licensed.
#' - Model export in plain Java code for deployment in production environments.
#' 
#' ## 
