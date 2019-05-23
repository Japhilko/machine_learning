#' ---
#' title: "Machine Learning with R - part 1"
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
## ---- include=FALSE------------------------------------------------------
knitr::opts_chunk$set(echo = FALSE,message = F,warning=F)

#' 
#' # Introduction to R
#' 
## ----child = 'a1_intro_r.Rmd'--------------------------------------------

#' 
## ----setupintror, include=FALSE------------------------------------------
knitr::opts_chunk$set(echo = T,cache=T,message=F,warning=F)
library(knitr)

#' 
#' ## Introduction round
#' 
#' ### Please tell us shortly...
#' 
#' - Where are you from? What are you studying/working?
#' - What is your experience level in R/other programming languages?
#' - What are your expectations of this course?
#' - Where do you think you can use Machine Learning in the future?
#' 
#' ## Preliminaries
#' 
#' - This topic is huge - we concentrate on presenting the applications in R
#' - Usually we have big differences in knowledge and abilities of the participants - please tell, if it is too fast or slow.
#' - We have many [**exercises**](http://web.math.ku.dk/~helle/R-intro/exercises.pdf) because at the end you can only learn on your own
#' - We have many [**examples**](https://www.showmeshiny.com/) - try them!
#' - If there are questions - always ask
#' - R is more fun together - ask your neighbor
#' 
#' 
#' ## Content of this section
#' 
#' - The first section is about laying the foundations in R. We will need all things covered later on.
#' 
#' ### Topics section:
#' 
#' - Why R is a good choice
#' - Constraints of R-usage
#' - R is modular 
#' - Import and export of data
#' 
#' <!--
#' - The second section is an introduction to the field of machine learning.
#' - The third part is on regression and classification.
#' -->
#' 
#' ## Why R is a good choice ...
#' 
#' - ... because it is an [**open source language**](https://stackoverflow.com/questions/1546583/what-is-the-definition-of-an-open-source-programming-language)
#' - ... outstanding graphs - [**graphics**](http://matthewlincoln.net/2014/12/20/adjacency-matrix-plots-with-r-and-ggplot2.html), [**graphics**](https://www.r-bloggers.com/3d-plots-with-ggplot2-and-plotly /), [**graphics**](https://procomun.wordpress.com/2011/03/18/splomr/)
#' - ... relates to other languages - [**R can be used in combination with other programs**](https://github.com/Japhilko/RInterfaces) - e.g. [**data linking**](https://github.com/Japhilko/RInterfaces/blob/master/slides/Datenimport.md)
#' - ...R can be used [**for automation**](https://cran.r-project.org/web/packages/MplusAutomation/index.html)
#'  - ... Vast Community - [**you can use the intelligence of other people ;-)**](https://www.r-bloggers.com/) and new statistical methodologies are implemented quite fast 
#' - Because R can be combined with other programs like `PostgreSQL` or `Python`
#' 
#' ## Constraints
#' 
#' <!--
#' Why R might not be the best choice in every situation
#' -->
#' 
#' <!--
#' https://elitedatascience.com/r-vs-python-for-data-science
#' -->
#' 
#' ### [Newer modules in Python](https://blog.dominodatalab.com/video-huge-debate-r-vs-python-data-science/)
#' 
#' - Machine learning is a field that changes rapidly.  
#' - Some new tools are first developed in Python. 
#' - The package `reticulate` offers the possibility to use these modules from an R environment. 
#' - Good news - Python is also Open Source
#' 
#' ### Big Data 
#' 
#' - Especially if you work with web data, you quickly have to deal with large amounts of data. 
#' - Therefore one must fall back on databases and parallelization strategies, which can be used in R. 
#' 
#' <!--
#' ## Content of this part
#' 
#' - Introduction to programming in R 
#' 
#' ### what is relevant for this course.
#' 
#' - How to import data?
#' - What to do with missing values?
#' - 
#' -->
#' 
#' ## R is modular
#' 
#' ### Install packages from CRAN Server
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("lme4")

#' 
#' ### Install packages from Bioconductor Server
#' 
## ----eval=F--------------------------------------------------------------
## source("https://bioconductor.org/biocLite.R")
## biocLite(c("GenomicFeatures", "AnnotationDbi"))

#' 
#' 
#' 
#' ### Install packages from Github
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("devtools")
## library(devtools)
## 
## devtools::install_github("koalaverse/vip")

#' 
#' <!--
#' https://github.com/koalaverse/vip
#' -->
#' 
#' 
#' ## [Task View Machine Learning](https://cran.r-project.org/web/views/MachineLearning.html)
#' 
#' 
#' ![](figure/taskviewmachinelearning.PNG){ width=110% }
#' 
#' 
#' 
#' ## Install all packages of a task view
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("ctv")
## ctv::install.views("MachineLearning")

#' 
#' 
#' ## Task: Find R-packages
#' 
#' Go to https://cran.r-project.org/ and search for packages that can be used:
#' 
#' - to reduce overfitting
#' - for random forests
#' - for gradient boosting
#' - for neural networks
#' - for clustering
#' 
#' 
#' <!--
#' https://www.r-bloggers.com/what-are-the-best-machine-learning-packages-in-r/
#' -->
#' 
#' 
#' ## Preparation - packages
#' 
## ------------------------------------------------------------------------
library(dplyr)

#' 
#' ![](figure/dplyr_vignette.PNG)
#' 
## ------------------------------------------------------------------------
library(magrittr)

#' 
#' ![](figure/magrittr_vignette.jpg)
#' ## Import `.csv` data 
#' 
#' ### The `read.csv` command
#' 
#' - Use `read.csv2` for German data 
#' 
## ----eval=F--------------------------------------------------------------
## ?read.csv
## ?read.csv2

#' 
#' ### Using a path to import data
#' 
## ----eval=F--------------------------------------------------------------
## path1<-"https://raw.githubusercontent.com/"
## path2<- "thomaspernet/data_csv_r/master/data/"
## dname <- "titanic_csv.csv"
## titanic <- read.csv(paste0(path1,path2,dname))

#' 
#' ### Save the dataset
#' 
## ----eval=F--------------------------------------------------------------
## save(titanic,file="../data/titanic.RData")

#' 
## ----echo=F--------------------------------------------------------------
load("../data/titanic.RData")

#' 
#' 
#' ## The titanic dataset
#' 
## ----eval=F,echo=F-------------------------------------------------------
## kable(head(titanic))

#' 
#' ![](figure/titanicdata.PNG)
#' <!--
#' https://www.guru99.com/r-decision-trees.html
#' -->
#' 
#' 
#' ## The function `scan` to import data 
#' 
#' - `scan` has an easy way to distinguish comments from data
#' 
## ----eval=F--------------------------------------------------------------
## ?scan

#' 
#' ### Example dataset
#' 
## ------------------------------------------------------------------------
cat("TITLE extra line", "# a comment","2 3 5 7", "11 13 17", 
    file = "../data/ex.data", sep = "\n")

#' 
#' ### Import data and skip the first line
#' 
## ----eval=F--------------------------------------------------------------
## pp<-scan("../data/ex.data",skip=1,quiet=TRUE)

#' 
## ------------------------------------------------------------------------
pp <- scan("../data/ex.data",comment.char="#", skip = 1,quiet = TRUE)

#' 
#' 
#' 
#' 
#' ## The download the data from UCI.
#' 
## ----eval=F,echo=F-------------------------------------------------------
## install.packages("bindrcpp")

#' 
## ------------------------------------------------------------------------
path1 <- "http://archive.ics.uci.edu/ml/"
path2 <- "machine-learning-databases/00243/"
dname <- 'yacht_hydrodynamics.data'

#' 
#' 
## ------------------------------------------------------------------------
url<- paste0(path1,path2,dname)
Yacht_Data <- readr::read_table(file = url)

#' 
## ----eval=F,echo=F-------------------------------------------------------
## colnam <- c('LongPos_COB', 'Prismatic_Coeff','Len_Disp_Ratio',
##             'Beam_Draut_Ratio','Length_Beam_Ratio','Froude_Num',
##             'Residuary_Resist')
## Yacht_Data <- read_table(file = url,col_names = colnam)

#' 
#' 
#' 
#' ## Built in datasets
#' 
#' - A sample dataset is often provided to demonstrate the functionality of a package.
#' - These records can be loaded using the `data` command.
#' 
## ------------------------------------------------------------------------
data(iris)

#' 
#' - There is also a [**RStudio Add-In**](https://github.com/bquast/datasets.load) that helps to find a built-in dataset.
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("datasets.load")

#' 
#' 
#' ## Exkurs [RStudio Addins](https://cran.r-project.org/web/packages/addinslist/README.html)
#' 
#' - Oben rechts befindet sich ein Button Addins 
#' 
#' ![](figure/addins.PNG)
#' 
#' ![](figure/datasetsload.PNG)
#' 
#' 
#' 
#' 
#' ## Exercise
#' 
#' Load the the built-in dataset `swiss` and answer the following questions:
#' 
#' - How many observations and variables are available?
#' - What is the scale level of the variables?
#' 
#' Create an interactive data table
#' 
#' 
#' 
#' ## The R-package `data.table`
#' 
#' ### Get an overview
#' 
## ------------------------------------------------------------------------
data(airquality)
head(airquality)

#' 
#' ## Overview with `data.table`
#' 
## ------------------------------------------------------------------------
library(data.table)
(airq <- data.table(airquality))

#' 
#' 
#' ## How to get help
#' 
#' -  I use [**duckduckgo:**](figure/duckduckgo.PNG)
#' 
#' ```
#' R-project + "what I want to know" 
#' ```
#' -  this works of course for all search engines!
#' 
#' 
#' ![](figure/duckduckgo.PNG)
#' 
#' 
#' ## [Exercise](https://www.datacamp.com/community/tutorials/pipe-r-tutorial)
#' 
## ----echo=F--------------------------------------------------------------
x <- c(0.109, 0.359, 0.63, 0.996, 0.515, 0.142, 0.017, 
       0.829, 0.907)
x <- runif(8)

#' 
#' - Draw 8 random numbers from the uniform distribution and save them in a vector `x`
#' - Compute the logarithm of `x`, return suitably lagged and iterated differences, 
#' - compute the exponential function and round the result
#' 
## ----echo=F--------------------------------------------------------------
round(exp(diff(log(x))), 1)

#' 
#' 
#' ## [The pipe operator](https://www.datacamp.com/community/tutorials/pipe-r-tutorial)
#' 
#' 
## ------------------------------------------------------------------------
library(magrittr)

# Perform the same computations on `x` as above
x %>% log() %>%
    diff() %>%
    exp() %>%
    round(1)

#' 
#' 
#' ## How to deal with missing values
#' 
## ----eval=F--------------------------------------------------------------
## ?na.omit

#' 
## ------------------------------------------------------------------------
airq

#' 
#' 
#' ## The command `na.omit`
#' 
## ------------------------------------------------------------------------
na.omit(airq)

#' 
#' 
#' ## [Clean the titanic data set](https://www.guru99.com/r-decision-trees.html)
#' 
## ------------------------------------------------------------------------
clean_titanic <- titanic %>% 	
  mutate(pclass=factor(pclass,levels = c(1, 2, 3),
                       labels=c('Upper','Middle','Lower')),
	survived = factor(survived,levels = c(0, 1), 
	                  labels=c('No', 'Yes'))) %>%
na.omit()

#' 
#' ###   `mutate(pclass = factor(...`: 
#' 
#' - Add label to the variable pclass. 
#' - 1 becomes Upper, 2 becomes MIddle and 3 becomes lower
#' 
#' ###  `factor(survived,...`:
#' 
#' - Add label to the variable survived. 
#' - 1 Becomes No and 2 becomes Yes
#' 
#' - `na.omit()`: Remove the NA observations 
#' 
#' 
#' ## Get an overview of the data
#' 
## ------------------------------------------------------------------------
glimpse(clean_titanic)

#' 
#' 
#' ## [Example Data - Housing Values in Suburbs of Boston](https://datascienceplus.com/fitting-neural-network-in-r/)
#' 
## ------------------------------------------------------------------------
library(MASS)
bdat <- Boston

#' 
## ----echo=F,eval=F-------------------------------------------------------
## kable(head(bdat))

#' 
#' ![](figure/bostondata.PNG)
#' 
#' ## Normalize your data
#' 
#' ### Compute maximum and minimum per column
#' 
## ------------------------------------------------------------------------
maxs <- apply(bdat, 2, max) 
mins <- apply(bdat, 2, min)

#' 
#' ### `scale` - Scaling and Centering of Matrix-like Objects
## ------------------------------------------------------------------------
scaled <- as.data.frame(scale(bdat, center = mins, 
                              scale = maxs - mins))


#' 
#' 
#' ## The scaled data
#' 
#' ![](figure/bostonscaled.PNG)
#' 
#' ## The command `sample`
#' 
#' - We can use this command to draw a sample. 
#' - We need the command later to split our dataset into a test and a training dataset. 
#' 
## ------------------------------------------------------------------------
sample(1:10,3,replace=T)
sample(1:10,3,replace=T)

#' 
#' 
#' ## Set a seed
#' 
#' - `set.seed` is the recommended way to specify seeds.
#' - If we set a seed, we get the same result for random events.
#' - This function is mainly required for simulations. 
#' 
## ------------------------------------------------------------------------
set.seed(234)
sample(1:10,3,replace=T)
set.seed(234)
sample(1:10,3,replace=T)

#' 
#' ## [Time measurement](https://www.r-bloggers.com/5-ways-to-measure-running-time-of-r-code/)
#' 
## ------------------------------------------------------------------------
start_time <- Sys.time()
ab <- runif(10000000)
end_time <- Sys.time()

end_time - start_time

#' 
#' 
#' ## How many cores are available
#' 
#' 
## ------------------------------------------------------------------------
library(doParallel)
detectCores()

#' 
#' ## Make cluster
#' 
## ------------------------------------------------------------------------
cl <- makeCluster(detectCores())
registerDoParallel(cl)

#' 
## ------------------------------------------------------------------------
start_time <- Sys.time()
ab <- runif(10000000)
end_time <- Sys.time()

end_time - start_time

#' 
## ------------------------------------------------------------------------
stopCluster(cl)

#' 
#' 
## ----eval=F--------------------------------------------------------------
## ?parallel::makeCluster

#' 
#' <!--
#' ## [The `swirl` package](https://swirlstats.com/)
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("swirl")

#' 
## ----eval=F--------------------------------------------------------------
## library("swirl")
## swirl()

#' -->
#' 
#' ## Resources
#' 
#' - [**Course materials for the Data Science Specialization**](https://github.com/DataScienceSpecialization/courses)
#' - Data wrangling - [**`dplyr` vignette**](https://cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html) - 
#' - The usage of pipes - [**`magrittr` vignette**](https://cran.r-project.org/web/packages/magrittr/vignettes/magrittr.html)
#' <!--
#' Further possible topics of this section:
#' 
#' - regular expressions
#' - the reticulate package
#' - how to install python modules
#' 
#' - mutate function in dplyr package
#' - the rescale package
#' -->
#' 
#' 

#' 
#' # Introduction to machine learning
#' 
## ----child = 'a2_intro_ml.Rmd'-------------------------------------------

#' 
## ----setupMlintro, include=FALSE-----------------------------------------
knitr::opts_chunk$set(echo = TRUE,cache=T)

#' 
#' ## [What Is Machine Learning?](https://www.netapp.com/us/info/what-is-machine-learning-ml.aspx)
#' 
#' - Machine learning allows the user to feed a computer algorithm an immense amount of data and have the computer analyze and make data-driven recommendations and decisions based on only the input data. 
#' 
#' - If any corrections are identified, the algorithm can incorporate that information to improve its future decision making.
#' 
#' <!--
#' 
#' ## Outline for today
#' 
#' - We start with supervised learning - we want to predict sale prices of flats/houses in Iowa using linear regression (part b1).
#' - In a next step we try to reduce overfitting using regularization methods (b2).
#' - 
#' 
#' -->
#' 
#' 
#' <!--
#' ## [Modern Machine Learning Algorithms](https://elitedatascience.com/machine-learning-algorithms)
#' -->
#' 
#' ### Categorizing machine learning algorithms...
#' 
#' - ... is tricky, and there are several approaches; 
#' - they can be grouped into generative/discriminative, parametric/non-parametric, supervised/unsupervised, and so on.
#' 
#' 
#' <!--
#' https://lgatto.github.io/IntroMachineLearningWithR/an-introduction-to-machine-learning-with-r.html
#' -->
#' 
#' 
#' ## [What is supervised learning?](https://elitedatascience.com/birds-eye-view)
#' 
#' 
#' <!--
#' https://lgatto.github.io/IntroMachineLearningWithR/supervised-learning.html#random-forest
#' -->
#' 
#' 
#' Supervised learning includes tasks for "labeled" data (i.e. you have a target variable).
#' 
#' <<<<<<< HEAD
#' - dimensionality refers to the number of features (i.e. input variables) 
#' 
#' - When the number of features is very large relative to the number of observations in your dataset, certain algorithms struggle to train effective models. This is called the “Curse of Dimensionality,” and it’s especially relevant for clustering algorithms that rely on distance calculations.
#' Dimensionality Reduciton
#' =======
#' - In practice, it's often used as an advanced form of predictive modeling.
#'  -  Each observation must be labeled with a "correct answer."
#'  -   Only then can you build a predictive model because you must tell the algorithm what's "correct" while training it (hence, "supervising" it).
#'  -   Regression is the task for modeling continuous target variables.
#'  -   Classification is the task for modeling categorical (a.k.a. "class") target variables.
#' 
#' >>>>>>> 88182c1f93446f7b0fa7a9c785df6515443ddecb
#' 
#' 
#' ## [Supervised vs unsupervised learning](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)
#' 
#' ### Supervised learning 
#' 
#' - Prior knowledge of what output values for samples should be. 
#' 
#' ![](figure/classification_regression.png){ height=20% }
#' 
#' ### Unsupervised learning
#' 
#' - Here the most common tasks are clustering, representation learning, and density estimation - we wish to learn the inherent structure of our data without using explicitly-provided labels. 
#' 
#' ![](figure/unsupervisedLearning.png){ height=20% }
#' 
#' <!--
#' https://medium.com/@danil.s.mikhailov/ai-and-the-social-sciences-part-i-5f172492d61d
#' -->
#' 
#' <!--
#' [Classification vs. regression ](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)
#' -->
#' 
#' 
#' 
#' ## [Machine Learning - Components](https://www.linkedin.com/pulse/20140822073217-180198720-6-components-of-a-machine-learning-algorithm)
#' 
#' - Feature Extraction + Domain knowledge 
#' 
#' <!--
#' (Import and Data Wrangling)
#' -->
#' 
#' - Feature Selection
#' 
#' - Choice of Algorithm (Regression or classification, regularization, decision trees, k-Means clustering, ...)
#' <!--
#' Naive Bayes
#' 
#' [Support Vector Machines](https://github.com/Japhilko/DataAnalysis/blob/master/Machine%20Learning/SupportVectorMachines.md)
#' -->
#' 
#' - Training
#' 
#' - Choice of Metrics/Evaluation Criteria
#' 
#' - Testing
#' 
#' 
#' ## [Feature selection in machine learning,... ](https://en.wikipedia.org/wiki/Feature_selection)
#' 
#' - ... is the process of selecting a subset of relevant features (variables, predictors) for use in model construction.
#' 
#' ### Four reasons for feature selection:
#' 
#' 1.) simplification of models to make them easier to interpret by researchers/users,
#' 
#' 2.) shorter training times,
#' 
#' 3.) to avoid the curse of dimensionality,
#' 
#' 4.) enhanced generalization by reducing overfitting (formally, reduction of variance)
#' 
#' 
#' 
#' ## [The Curse of Dimensionality](https://elitedatascience.com/dimensionality-reduction-algorithms)
#' 
#' ![](figure/3d-coordinate-plane.png){ height=40% }
#' 
#' In machine learning, “dimensionality” simply refers to the number of features (i.e. input variables) in your dataset.
#' 
#' - When the number of features is very large relative to the number of observations, certain algorithms struggle to train effective models. 
#' 
#' - This is called the “Curse of Dimensionality,” and it’s especially relevant for clustering algorithms that rely on distance calculations.
#' 
#' 
#' 
#' ## [What are the advantages and disadvantages of decision trees?](https://elitedatascience.com/machine-learning-interview-questions-answers#supervised-learning)
#' 
#' Advantages: Decision trees are easy to interpret, nonparametric (which means they are robust to outliers), and there are relatively few parameters to tune.
#' 
#' Disadvantages: Decision trees are prone to be overfit. 
#' 
#' - This can be addressed by ensemble methods like random forests or boosted trees.
#' 
#' 
#' 
#' <!--
#' - Example data are used to train a model.
#' - With this model the classification can be realized automatically. 
#' 
#' 
#' http://www.datenbanken-verstehen.de/lexikon/supervised-learning/
#' 
#' Daten einer Gruppierung zuzuordnen, die durch den Nutzenden vorgegeben sind, aber nicht jeder Datensatz manuell bewertet werden kann (z. B. Kreditbewilligung abhängig von Kredithöhe und Bonität). 
#' 
#' Die Aufgabe besteht darin, 
#' 
#' Ein Modell wird mit Beispieldaten aufgebaut, das die Zuordnung anschließend selbstständig übernimmt.
#' -->
#' 
#' 
#' 
#' ## [Random Forest](https://www.datascience.com/resources/notebooks/random-forest-intro)
#' 
#' > Random forest aims to reduce the previously mentioned correlation issue by choosing only a subsample of the feature space at each split. Essentially, it aims to make the trees de-correlated and prune the trees by setting a stopping criteria for node splits, which I will cover in more detail later.
#' 
#' 
#' ## [Random forest](https://en.wikipedia.org/wiki/Random_forest)
#' 
#' - Ensemble learning method - multitude of decision trees 
#' - Random forests correct for decision trees' habit of overfitting to their training set.
#' 
#' ![](figure/expl_rf.png)
#' 
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
#' attempts to reduce the chance overfitting complex models.
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
#' 
#' 
#' 
#' ## [Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
#' 
#' Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.
#' 
#' The idea of gradient boosting originated in the observation by Leo Breiman that boosting can be interpreted as an optimization algorithm on a suitable cost function.
#' 
#' 
#' Breiman, L. (1997). "Arcing The Edge". Technical Report 486. Statistics Department, University of California, Berkeley.
#' 
#' 
#' <!--
#' -->
#' 
#' 
#' <!--
#' ## Explicit algorithms
#' 
#' Explicit regression gradient boosting algorithms were subsequently developed by Jerome H. Friedman, simultaneously with the more general functional gradient boosting perspective of Llew Mason, Jonathan Baxter, Peter Bartlett and Marcus Frean.
#' 
#' 
#' The latter two papers introduced the view of boosting algorithms as iterative functional gradient descent algorithms. That is, algorithms that optimize a cost function over function space by iteratively choosing a function (weak hypothesis) that points in the negative gradient direction. This functional gradient view of boosting has led to the development of boosting algorithms in many areas of machine learning and statistics beyond regression and classification.
#' -->
#' 
#' ## [**Advantages of gradient boosting**](http://uc-r.github.io/gbm_regression)
#' 
#' - Often provides predictive accuracy that cannot be beat.
#' - Lots of flexibility - can optimize on different loss functions and provides several hyperparameter tuning options that make the function fit very flexible.
#' - No data pre-processing required - often works great with categorical and numerical values as is.
#' - Handles missing data - imputation not required.
#' 
#' ## [**Disadvantages**](http://uc-r.github.io/gbm_regression) of gradient boosting
#' 
#' 
#' - GBMs will continue improving to minimize all errors. This can overemphasize outliers and cause overfitting. Must use cross-validation to neutralize.
#' - Computationally expensive - GBMs often require many trees (>1000) which can be time and memory exhaustive.
#' - The high flexibility results in many parameters that interact and influence heavily the behavior of the approach (number of iterations, tree depth, regularization parameters, etc.). This requires a large grid search during tuning.
#' - Less interpretable although this is easily addressed with various tools (variable importance, partial dependence plots, LIME, etc.).
#' 
#' 
#' ## Two types of errors for tree methods
#' 
#' ### Bias related errors
#' 
#' - Adaptive boosting
#' - Gradient boosting
#' 
#' ### Variance related errors
#' 
#' - Bagging
#' - Random forest
#' 
#' <!--
#' https://www.slideshare.net/JaroslawSzymczak1/gradient-boosting-in-practice-a-deep-dive-into-xgboost
#' 
#' What if we, instead of reweighting examples, made some corrections to prediction errors directly?
#' 
#' Residual is a gradient of single observation error contribution in one of the most common evaluation measure for regression: RMSE
#' -->
#' 
#' 
#' <!--
#' ## [Gradient Boosting for Linear Regression - why does it not work?](https://stats.stackexchange.com/questions/186966/gradient-boosting-for-linear-regression-why-does-it-not-work)
#' 
#' 
#' While learning about Gradient Boosting, I haven't heard about any constraints regarding the properties of a "weak classifier" that the method uses to build and ensemble model. 
#' 
#' 
#' - I could not imagine an application of a GB that uses linear regression, and in fact when I've performed some tests - it doesn't work. I was testing the most standard approach with a gradient of sum of squared residuals and adding the subsequent models together.
#' 
#' The obvious problem is that the residuals from the first model are populated in such manner that there is really no regression line to fit anymore. My another observation is that a sum of subsequent linear regression models can be represented as a single regression model as well (adding all intercepts and corresponding coefficients) so I cannot imagine how that could ever improve the model. The last observation is that a linear regression (the most typical approach) is using sum of squared residuals as a loss function - the same one that GB is using.
#' 
#' I also thought about lowering the learning rate or using only a subset of predictors for each iteration, but that could still be summed up to a single model representation eventually, so I guess it would bring no improvement.
#' 
#' What am I missing here? Is linear regression somehow inappropriate to use with Gradient Boosting? Is it because the linear regression uses the sum of squared residuals as a loss function? Are there any particular constraints on the weak predictors so they can be applied to Gradient Boosting?
#' -->
#' 
#' 
#' ## Links and resources
#' 
#' - [Presentations on ‘Elements of Neural Networks & Deep Learning’ ](https://www.r-bloggers.com/my-presentations-on-elements-of-neural-networks-deep-learning-parts-45/)
#' 
#' - [Understanding the Magic of Neural Networks](https://www.r-bloggers.com/understanding-the-magic-of-neural-networks/)
#' 
#' <!--
#' - [Neural Text Modelling with R package ruimtehol](https://www.r-bloggers.com/neural-text-modelling-with-r-package-ruimtehol/)
#' -->
#' 
#' - [Feature Selection using Genetic Algorithms in R](https://www.r-bloggers.com/feature-selection-using-genetic-algorithms-in-r/)
#' 
#' - [Lecture slides: Real-World Data Science (Fraud Detection, Customer Churn & Predictive Maintenance)](https://www.r-bloggers.com/lecture-slides-real-world-data-science-fraud-detection-customer-churn-predictive-maintenance/)
#' 
#' - [Automated Dashboard for Credit Modelling with Decision trees and Random forests in R](https://www.r-bloggers.com/automated-dashboard-for-credit-modelling-with-decision-trees-and-random-forests-in-r/)
#' 
#' - [Looking Back at Google’s Research Efforts in 2018](https://ai.googleblog.com/2019/01/looking-back-at-googles-research.html)
#' 
#' - [Selecting ‘special’ photos on your phone](https://www.r-bloggers.com/selecting-special-photos-on-your-phone/)
#' 
#' 
#' - [Open Source AI, ML & Data Science News](https://www.r-bloggers.com/ai-machine-learning-and-data-science-roundup-january-2019/)
#' <!--
#' Datacamp Course
#' 
#' https://www.r-bloggers.com/my-course-on-hyperparameter-tuning-in-r-is-now-on-data-camp/
#' 
#' company quantide
#' 
#' 
#' https://medium.freecodecamp.org/every-single-machine-learning-course-on-the-internet-ranked-by-your-reviews-3c4a7b8026c0
#' -->
#' 
#' - Google`s [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/)
#' 
#' - [A prelude to machine learning](https://eight2late.wordpress.com/2017/02/23/a-prelude-to-machine-learning/)
#' 
#' - [`caret` webinar by Max Kuhn - on youtube](https://www.youtube.com/watch?v=7Jbb2ItbTC4)
#' 
#' - [Learn math for data science](https://elitedatascience.com/learn-math-for-data-science)
#' - [Learn statistics for data science](https://elitedatascience.com/learn-statistics-for-data-science)
#' 
#' - [Aachine learning projects for beginners](https://elitedatascience.com/machine-learning-projects-for-beginners)
#' 
#' ## Links and resources (II)
#' 
#' - [An Introduction to machine learning](http://www-bcf.usc.edu/~gareth/ISL/)
#' - [ISLR book](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf)
#' 
#' - [**useR! Machine Learning Tutorial**](https://koalaverse.github.io/machine-learning-in-R/)
#' 
#' ### Introduction to machine learning with R
#' 
#' - [Your First Machine Learning Project in R Step-By-Step](https://machinelearningmastery.com/machine-learning-in-r-step-by-step/)
#' 
#' 
#' - chapter about machine learning in [awesome R](https://awesome-r.com/)
#' 
#' 
#' - [Shiny App for machine learning](https://www.showmeshiny.com/machlearn/)
#' 
#' 
#' 
#' ## Annex - [Prediction vs. Causation in Regression Analysis](https://statisticalhorizons.com/prediction-vs-causation-in-regression-analysis)
#' 
#' ### [**Paul Allison**](https://statisticalhorizons.com/prediction-vs-causation-in-regression-analysis)
#' 
#' > There are two main uses of multiple regression: prediction and causal analysis. In a prediction study, the goal is to develop a formula for making predictions about the dependent variable, based on the observed values of the independent variables….In a causal analysis, the independent variables are regarded as causes of the dependent variable. The aim of the study is to determine whether a particular independent variable really affects the dependent variable, and to estimate the magnitude of that effect, if any
#' 
#' <!--
#' ## Literature for machine learning
#' 
#' ![](figure/book_ml1.jpg)
#' 
#' -->
#' <!--
#' https://lgatto.github.io/IntroMachineLearningWithR/index.html
#' https://www.kaggle.com/camnugent/introduction-to-machine-learning-in-r-tutorial
#' 
#' https://www.r-bloggers.com/in-depth-introduction-to-machine-learning-in-15-hours-of-expert-videos/
#' 
#' https://www.r-bloggers.com/my-presentations-on-elements-of-neural-networks-deep-learning-parts-678/
#' -->

#' 
#' # Simple regression
#' 
## ----child = 'b1_regression.Rmd'-----------------------------------------

#' 
## ----setupregression, include=FALSE--------------------------------------
knitr::opts_chunk$set(echo = T,warning = F,message = F)
pres=T

#' 
#' ## Why a part on simple regression
#' 
#' - Some machine learning concepts are based on regression
#' - We would like to remind you how simple regression works in R. 
#' - We also want to show the constraints
#' - In a next step we will learn, how to coop with these constraints
#' 
#' <!--
#' http://enhancedatascience.com/2017/06/29/machine-learning-explained-overfitting/
#' -->
#' 
#' ## Variables of the `mtcars` dataset
#' 
#' Help for the `mtcars` dataset:
#' 
## ----eval=F--------------------------------------------------------------
## ?mtcars

#' <!--
#' displacement - Hubraum
#' -->
#' -	 mpg -	 Miles/(US) gallon
#' -	 cyl -	 Number of cylinders
#' -	 disp	- Displacement (cu.in.)
#' -	 hp	- Gross horsepower
#' -	 drat -	 Rear axle ratio
#' - wt	- Weight (1000 lbs)
#' -	 qsec	- 1/4 mile time
#' -	 vs	- Engine (0 = V-shaped, 1 = straight)
#' -	 am	-  Transmission (0 = automatic, 1 = manual)
#' -	 gear	-  Number of forward gears
#' -	 carb -	 Number of carburetors
#' 
#' 
#' ## Dataset `mtcars`
#' 
## ----echo=F--------------------------------------------------------------
library(knitr)
kable(mtcars)

#' 
#' 
#' ## Distributions of two variables of `mtcars`
#' 
## ------------------------------------------------------------------------
par(mfrow=c(1,2))
plot(density(mtcars$wt)); plot(density(mtcars$mpg))

#' 
#' 
#' 
#' ## A simple regression model
#' 
#' ### Dependent variable - miles per gallon (mpg)
#' 
#' ### Independent variable - weight (wt)
#' 
## ------------------------------------------------------------------------
m1 <- lm(mpg ~ wt,data=mtcars)
m1

#' 
#' ## Get the model summary 
#' 
## ------------------------------------------------------------------------
summary(m1) 

#' 
#' ## The model formula
#' 
#' ### Model without intercept
#' 
## ------------------------------------------------------------------------
m2 <- lm(mpg ~ - 1 + wt,data=mtcars)
summary(m2)$coefficients

#' 
#' ### Adding further variables
#' 
## ------------------------------------------------------------------------
m3 <- lm(mpg ~ wt + cyl,data=mtcars)
summary(m3)$coefficients

#' 
#' ## The command `as.formula`
#' 
## ----eval=F--------------------------------------------------------------
## ?as.formula

#' 
#' 
## ------------------------------------------------------------------------
class(fo <- mpg ~ wt + cyl)

#' 
## ------------------------------------------------------------------------
# The formula object can be used in the regression:
m3 <- lm(fo,data=mtcars)

#' 
#' 
#' ## [Further possibilities to specify the formula](https://cran.r-project.org/web/packages/Formula/vignettes/Formula.pdf)
#' 
#' ### Interaction effect
#' 
## ------------------------------------------------------------------------
# effect of cyl and interaction effect:
m3a<-lm(mpg~wt*cyl,data=mtcars) 

# only interaction effect:
m3b<-lm(mpg~wt:cyl,data=mtcars) 

#' 
#' 
## ----eval=F,echo=F-------------------------------------------------------
## m3c<-lm(mpg~cyl|wt,data=mtcars)
## 
## m3c<-lm(mpg~cyl/wt,data=mtcars)
## 
## 
## summary(m3b)
## summary(m3c)

#' 
#' 
#' ### Take the logarithm
#' 
## ------------------------------------------------------------------------
m3d<-lm(mpg~log(wt),data=mtcars) 

#' 
#' 
#' <!--
#' https://www.r-bloggers.com/r-tutorial-series-regression-with-interaction-variables/
#' 
#' https://www.r-bloggers.com/interpreting-interaction-coefficient-in-r-part1-lm/
#' -->
#' 
#' ## The command `model.matrix`
#' 
#' <!--
#' - Construct Design Matrices
#' 
#' https://genomicsclass.github.io/book/pages/expressing_design_formula.html
#' -->
#' 
#' - With `model.matrix`the qualitative variables are automatically dummy encoded 
#' 
## ----eval=F--------------------------------------------------------------
## ?model.matrix

#' 
#' 
## ------------------------------------------------------------------------
model.matrix(m3d)

#' 
#' 
#' ## Model matrix (II)
#' 
#' - [We can also create a model matrix directly from the formula and data arguments](http://pages.stat.wisc.edu/~st849-1/Rnotes/ModelMatrices.html)
#' - See `Matrix::sparse.model.matrix` for increased efficiency on large dimension data.
#' 
## ------------------------------------------------------------------------
ff <- mpg ~ log(wt):cyl
m <- model.frame(ff, mtcars)

#' 
## ------------------------------------------------------------------------
(mat <- model.matrix(ff, m))

#' 
#' 
#' <!--
#' m3c <- lm(y = mtcars$mpg,x=mat[,-1])
#' -->
#' 
#'  
#' 
#' 
#' ## A model with interaction effect
#' 
#' <!--
#' drat - Hinterachsenübersetzung
#' disp - Hubraum
#' -->
#' 
## ------------------------------------------------------------------------
# disp	-  Displacement (cu.in.)
m3d<-lm(mpg~wt*disp,data=mtcars) 
m3dsum <- summary(m3d)
m3dsum$coefficients

#' 
#' 
#' <!--
#' ## [Exploring interactions](https://cran.r-project.org/web/packages/jtools/vignettes/interactions.html)
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("jtools")

#' 
## ----eval=F,echo=T-------------------------------------------------------
## library(jtools)
## interact_plot(m3d, pred = "wt", modx = "disp")

#' 
#' - With a continuous moderator (in our case `disp`) you get three lines — 1 standard deviation above and below the mean and the mean itself. 
#' 
#' ![](figure/mtcars_model_interact.PNG)
#' 
## ----eval=F,echo=F-------------------------------------------------------
## library(jtools)
## fitiris <- lm(Petal.Length ~ Petal.Width * Species, data = iris)
## interact_plot(fitiris, pred = "Petal.Width", modx = "Species")

#' -->
#' 
#' ## Residual plot - model assumptions violated? 
#' 
#' 
#' - We have model assumptions violated if points deviate with a pattern from the line 
#' 

#' 
#' ![](figure/resid_fitted.PNG)
#' 
#' ## Residual plot
#' 
## ------------------------------------------------------------------------
plot(m3,2)

#' 
#' - If the residuals are normally distributed, they should be on the same line.
#' 
#' 
#' ## Example: object orientation
#' 
#' - `m3` is now a special regression object
#' - Various functions can be applied to this object
#' 
## ----eval=F--------------------------------------------------------------
## predict(m3) # Prediction
## resid(m3) # Residuals

#' 
## ----echo=F--------------------------------------------------------------
head(predict(m3)) # Prediction
head(resid(m3)) # Residuals

#' 
#' 
#' ## Make model prediction
#' 
## ------------------------------------------------------------------------
pre <- predict(m1)
head(mtcars$mpg)
head(pre)

#' 
#' ## Regression diagnostic with base-R
#' 
## ----eval=F--------------------------------------------------------------
## plot(mtcars$wt,mtcars$mpg)
## abline(m1)
## segments(mtcars$wt, mtcars$mpg, mtcars$wt, pre, col="red")

#' 
#' ![](figure/prediction_mtcars.PNG)
#' 
## ----echo=F,eval=F-------------------------------------------------------
## # https://www.r-bloggers.com/marginal-effects-for-regression-models-in-r-rstats-dataviz/
## p <- ggpredict(m5, c("wt", "cyl"))
## plot(p)

#' 
#' ## The mean squared error
#' 
#' - The [**MSE**](https://en.wikipedia.org/wiki/Mean_squared_error) measures the average of the squares of the errors
#' - [**The lower the better**](http://r-statistics.co/Linear-Regression.html)
#' 
## ------------------------------------------------------------------------
(mse5 <- mean((mtcars$mpg -  pre)^2)) # model 5
(mse4 <- mean((mtcars$mpg -  predict(m4))^2)) # model 4

#' 
#' 
#' 
#' ## The `visreg`-package
#' 
## ----eval=F--------------------------------------------------------------
## install.packages("visreg")

#' 
## ------------------------------------------------------------------------
library(visreg)

#' 
#' ![](figure/visreg.PNG)
#' 
#' ## The `visreg`-package
#' 
#' - The default-argument for `type` is `conditional`.
#' - Scatterplot of `mpg` and `wt` plus regression line and confidence bands
#' 
## ----eval=F--------------------------------------------------------------
## visreg(m1, "wt", type = "conditional")

#' 
## ----eval=F,echo=F-------------------------------------------------------
## visreg(m1, "wt", type = "conditional",
##       line=list(col="red"),
##        fill=list(col="#473C8B"),points=list(cex=1.5,col=rgb(0,1,0,.5)))

#' 
#' ![](figure/visregplot1.PNG)
#' 
#' 
#' <!--
#' ## [Visualisation with `visreg` ](http://myweb.uiowa.edu/pbreheny/publications/visreg.pdf)
#' 
#' - [Second argument](http://pbreheny.github.io/visreg) -  Specification covariate for visualisation
#' - plot shows the effect on the expected value of the response by moving the x variable away from a reference point on the x-axis (for numeric variables, the mean).
#' 
## ----eval=F--------------------------------------------------------------
## visreg(m1, "wt", type = "contrast")

#' 
#' 
## ----echo=F,eval=F-------------------------------------------------------
## visreg(m1, "wt", type = "contrast",alpha=.01,
##        line=list(col="red"),
##        fill=list(col="#473C8B"),points=list(cex=1.5,col=rgb(.4,.4,0,.5)))

#' 
#' ![](figure/visreg2.PNG)
#' -->
#' 
#' 
#' ## Regression with factors
#' 
#' - The effects of factors can also be visualized with `visreg`:
#' 
## ------------------------------------------------------------------------
mtcars$cyl <- as.factor(mtcars$cyl)
m4 <- lm(mpg ~ cyl + wt, data = mtcars)
# summary(m4)

#' 
## ----echo=F--------------------------------------------------------------
sum_m4 <- summary(m4)
sum_m4$coefficients

#' 
#' 
#' ## Effects of factors
#' 
#' 
## ----eval=F--------------------------------------------------------------
## par(mfrow=c(1,2))
## visreg(m4, "cyl", type = "contrast")
## visreg(m4, "cyl", type = "conditional")

#' 
## ----eval=F,echo=F-------------------------------------------------------
## par(mfrow=c(1,2))
## visreg(m4, "cyl", type = "contrast",fill=list(col=c("#00FFFF")),points=list(cex=1.5,col=rgb(.4,.4,.4,.5)))
## visreg(m4, "cyl", type = "conditional",fill=list(col=c("#00FFFF")),points=list(cex=1.5,col=rgb(.4,.4,.4,.5)))

#' 
#' ![](figure/visregcat.PNG)
#' 
#' <!--
#' ## The command `model.matrix`
#' 
## ----eval=F--------------------------------------------------------------
## ?model.matrix

#' -->
#' 
#' 
#' ## The package `visreg` - Interactions
#' 
## ------------------------------------------------------------------------
m5 <- lm(mpg ~ cyl*wt, data = mtcars)
# summary(m5)

#' 
## ----echo=F--------------------------------------------------------------
sum_m5 <- summary(m5)
sum_m5$coefficients

#' 
#' 
#' ## Control of the graphic output with `layout`.
#' 

#' 
#' 

#' 
#' ![](figure/factor3vars_visreg.PNG)
#' 
#' ## The package `visreg` - Interactions overlay
#' 
## ------------------------------------------------------------------------
m6 <- lm(mpg ~ hp + wt * cyl, data = mtcars)

#' 
#' 

#' 
#' ![](figure/visreg_m6.PNG)
#' 
#' ## The package `visreg` - `visreg2d`
#' 
## ------------------------------------------------------------------------
visreg2d(m6, "wt", "hp", plot.type = "image")

#' 
#' <!--
#' ## The package `visreg` - `surface`
#' 
## ------------------------------------------------------------------------
visreg2d(m6, "wt", "hp", plot.type = "persp")

#' -->
#' 
#' ## Nice table output with [`stargazer`](https://cran.r-project.org/web/packages/stargazer/vignettes/stargazer.pdf)
#' 
#' 
## ----eval=F,echo=F-------------------------------------------------------
## install.packages("stargazer")

#' 
## ----eval=F--------------------------------------------------------------
## library(stargazer)
## stargazer(m3, type="html")

#' 
#' ### Example HTML output:
#' 
#' ![](figure/stargazertabex.PNG)
#' 
#' ## Exercise
#' 
#' - Install the package `AmesHousing` and create a [**processed version**](https://cran.r-project.org/web/packages/AmesHousing/AmesHousing.pdf) of the Ames housing data with the variables `Sale_Price`, `Gr_Liv_Area` and `TotRms_AbvGrd`
#' - Create a regression model with `Sale_Price` as dependent and `Gr_Liv_Area` and `TotRms_AbvGrd` as independent variables. Then create seperated models for the two independent variables. Compare the results. What do you think?
#' 
#' <!--
#' lm(Sale_Price ~ Gr_Liv_Area + TotRms_AbvGrd, data = ames_data)
#' -->
#' 
#' ## [The Ames Iowa Housing Data](http://ww2.amstat.org/publications/jse)
#' 
## ------------------------------------------------------------------------
ames_data <- AmesHousing::make_ames()

#' 
#' ### Some Variables
#' 
#' - `Gr_Liv_Area`: Above grade (ground) living area square feet
#' - `TotRms_AbvGrd`: Total rooms above grade (does not include bathrooms
#' - `MS_SubClass`: Identifies the type of dwelling involved in the sale.
#' - `MS_Zoning`: Identifies the general zoning classification of the sale.
#' - `Lot_Frontage`: Linear feet of street connected to property
#' - `Lot_Area`: Lot size in square feet
#' - `Street`: Type of road access to property
#' - `Alley`: Type of alley access to property
#' - `Lot_Shape`: General shape of property
#' - `Land_Contour`: Flatness of the propert
#' 
#' 
#' ## Multicollinearity
#' 
#' - As p increases we are more likely to capture multiple features that have some multicollinearity. 
#' - When multicollinearity exists, we often see high variability in our coefficient terms. 
#' - E.g. we have a correlation of 0.801 between `Gr_Liv_Area` and `TotRms_AbvGrd` 
#' - Both variables are strongly correlated to the response variable (`Sale_Price`).
#' 
## ----echo=F--------------------------------------------------------------
library(AmesHousing) 
ames_data <- AmesHousing::make_ames()

#' 
#' 
## ------------------------------------------------------------------------
ames_data <- AmesHousing::make_ames()
cor(ames_data[,c("Sale_Price","Gr_Liv_Area","TotRms_AbvGrd")])

#' 
#' 
#' ## Multicollinearity
#' 
## ------------------------------------------------------------------------
lm(Sale_Price ~ Gr_Liv_Area + TotRms_AbvGrd, data = ames_data)

#' 
#' - When we fit a model with both these variables we get a positive coefficient for `Gr_Liv_Area` but a negative coefficient for `TotRms_AbvGrd`, suggesting one has a positive impact to Sale_Price and the other a negative impact.
#' 
#' ## Seperated models
#' 
#' - If we refit the model with each variable independently, they both show a positive impact. 
#' - The `Gr_Liv_Area` effect is now smaller and the `TotRms_AbvGrd` is positive with a much larger magnitude.
#' 
## ------------------------------------------------------------------------
lm(Sale_Price ~ Gr_Liv_Area, data = ames_data)$coefficients

#' 
## ------------------------------------------------------------------------
lm(Sale_Price ~ TotRms_AbvGrd, data = ames_data)$coefficients

#' 
#' - This is a common result when collinearity exists. 
#' - Coefficients for correlated features become over-inflated and can fluctuate significantly. 
#' 
#' 
#' ## Consequences
#' 
#' - One consequence of these large fluctuations in the coefficient terms is [**overfitting**](https://en.wikipedia.org/wiki/Overfitting), which means we have high variance in the bias-variance tradeoff space. 
#' - We can use tools such as [**variance inflaction factors**](https://en.wikipedia.org/wiki/Variance_inflation_factor) (Myers, 1994) to identify and remove those strongly correlated variables, but it is not always clear which variable(s) to remove. 
#' - Nor do we always wish to remove variables as this may be removing signal in our data.
#' 
#' 
#' 
#' ## The problem - [Overfitting](https://elitedatascience.com/overfitting-in-machine-learning#examples)
#' 
#' - Our model doesn’t generalize well from our training data to unseen data.
#' 
#' 
#' ![](figure/Overfitting_fig1.PNG)
#' 
#' <!--
#' ## [The Signal and the Noise](https://en.wikipedia.org/wiki/The_Signal_and_the_Noise)
#' 
#' - In predictive modeling, you can think of the “signal” as the true underlying pattern that you wish to learn from the data.
#' - “Noise,” on the other hand, refers to the irrelevant information or randomness in a dataset.
#' 
#' ![](figure/The_Signal_and_the_Noise.jpg)
#' -->
#' 
#' 
#' <!--
#' https://cran.r-project.org/web/packages/keras/vignettes/tutorial_basic_regression.html
#' -->
#' 
#' 
#' <!--
#' https://cran.r-project.org/web/packages/keras/vignettes/tutorial_overfit_underfit.html
#' 
#' https://www.r-bloggers.com/machine-learning-explained-overfitting/
#' -->
#' 
#' 
#' <!--
#' ![](figure/electoral_precedent.png)
#' -->
#' 
#' <!--
#' ## [Overfitting](https://en.wikipedia.org/wiki/Overfitting).
#' 
#' ![](figure/450px-Overfitting.svg.png)
#' 
#' The green line represents an overfitted model and the black line represents a regularized model. While the green line best follows the training data, it is too dependent on that data and it is likely to have a higher error rate on new unseen data, compared to the black line.
#' -->
#' 
#' <!--
#' https://en.wikipedia.org/wiki/Overfitting
#' -->
#' 
#' ## What can be done against overvitting
#' 
#' - Cross Validation 
#' - Train with more data
#' - Remove features
#' 
#' - Regularization - e.g. ridge and lasso regression
#' - Ensembling - e.g. bagging and boosting
#' 
#' <!--
#' ## [Cross-validation](https://elitedatascience.com/overfitting-in-machine-learning#examples)
#' 
#' - [**3 fold cross validation**](https://www.statmethods.net/stats/regression.html)
#' 
## ----eval=F,echo=F-------------------------------------------------------
## amod1 <- lm(Sale_Price ~ TotRms_AbvGrd, data = ames_data)
## # K-fold cross-validation
## library(DAAG)
## DAAG::cv.lm(data=ames_data,form.lm =  amod1, m=3) # 3 fold cross-validation

#' -->
#' 
#' 
#' 
#' ## Cross validation
#' 
#' - Cross-validation is a powerful preventative measure against overfitting.
#' 
#' - Use your initial training data to generate multiple mini train-test splits. Use these splits to tune your model.
#' 
#' 
#' ### Necessary packages
#' 
## ------------------------------------------------------------------------
library(tidyverse)
library(caret)

#' 
#' ### Swiss Fertility and Socioeconomic Indicators
#' 
## ------------------------------------------------------------------------
data("swiss")

#' 
#' ## [Cross Validation in R](http://www.sthda.com/english/articles/38-regression-model-validation/157-cross-validation-essentials-in-r/)
#' 
#' 
#' ### Split data into training and testing dataset
#' 
## ------------------------------------------------------------------------
training.samples <- swiss$Fertility %>%
createDataPartition(p = 0.8, list = FALSE)
train.data  <- swiss[training.samples, ]
test.data <- swiss[-training.samples, ]

#' 
#' ### Build the model and make predictions
#' 
## ------------------------------------------------------------------------
model <- lm(Fertility ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
(predictions <- model %>% predict(test.data))

#' 
#' ## Model with cross validation
#' 
#' - Loocv: [**leave one out cross validation**](https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/)
#' 
## ------------------------------------------------------------------------
train.control <- caret::trainControl(method = "LOOCV")
# Train the model
model <- train(Fertility ~., data = swiss, method = "lm",
               trControl = train.control)
model %>% predict(test.data)

#' 
#' <!--
#' ## [k-fold cross validation](https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/)
#' -->
#' 
#' ## Summarize the results
#' 
## ------------------------------------------------------------------------
print(model)

#' 
#' 
#' <!--
#' ## [The bias variance tradeoff](https://elitedatascience.com/bias-variance-tradeoff)
#' 
#' ![](figure/bias_variance_tradeoff.PNG){ height=70% }
#' -->
#' 
#' <!--
#' ## Good literature for linear regression in R
#' 
#' ### Useful PDF document:
#' 
#' J H Maindonald - [**Using R for Data Analysis and Graphics
#' Introduction, Code and Commentary**](https://cran.r-project.org/doc/contrib/usingR.pdf)
#' 
#' -  Introduction to R 
#' -  Data analysis
#' -  Statistical models
#' -  Inference concepts
#' -  Regression with one predictor
#' -  Multiple linear regression
#' -  Extending the linear model
#' -  ...
#' -->
#' 
#' 
#' <!--
#' Anwendung log Transformation
#' wann wird dies gemacht
#' -->
#' 
#' ## Links - linear regression
#' 
#' -  Regression - [**r-bloggers**](http://www.r-bloggers.com/r-tutorial-series-simple-linear-regression/)
#' 
#' -  The complete book of [**Faraway**](http://cran.r-project.org/doc/contrib/Faraway-PRA.pdf)- very intuitive
#' 
#' -  Good introduction on [**Quick-R**](http://www.statmethods.net/stats/regression.html)
#' 
#' - [**Multiple regression**](https://www.r-bloggers.com/multiple-regression-part-1/)
#' 
#' - [**15 Types of Regression you should know**](https://www.r-bloggers.com/15-types-of-regression-you-should-know/)
#' 
#' - [**`ggeffects` - Create Tidy Data Frames of Marginal Effects for ‘ggplot’ from Model Outputs**](https://strengejacke.github.io/ggeffects/)
#' 
#' 
#' - [**Machine learning iteration**](https://elitedatascience.com/machine-learning-iteration)
#' 
#' 
#' <!--
#' https://www.dataquest.io/blog/statistical-learning-for-predictive-modeling-r/
#' 
#' https://www.r-bloggers.com/example-of-overfitting/
#' 
#' https://blog.minitab.com/blog/adventures-in-statistics-2/the-danger-of-overfitting-regression-models
#' 
#' 
#' https://statisticsbyjim.com/regression/overfitting-regression-models/
#' 
#' https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765
#' 
#' https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/
#' 
#' 
#' https://statisticsbyjim.com/regression/overfitting-regression-models/
#' -->
#' 
#' 
#' 
#' 
#' 
#' ## Shiny App - Diagnostics for linear regression
#' 
#' <!--
#' https://gallery.shinyapps.io/slr_diag/
#' -->
#' 
#' - Shiny App - [**Simple Linear Regression**](https://gallery.shinyapps.io/simple_regression/)
#' 
#' - Shiny App - [**Multicollinearity in multiple regression**](figure/https://gallery.shinyapps.io/collinearity/)
#' 
#' 
#' [![](figure/Diagslr.PNG)](https://gallery.shinyapps.io/slr_diag/)
#' 
#' <!--
#' https://www.r-bloggers.com/elegant-regression-results-tables-and-plots-in-r-the-finalfit-package/
#' https://www.r-bloggers.com/regression-analysis-essentials-for-machine-learning/
#' https://www.r-bloggers.com/15-types-of-regression-you-should-know/
#' https://www.r-bloggers.com/marginal-effects-for-regression-models-in-r-rstats-dataviz/
#' http://pbreheny.github.io/visreg/contrast.html
#' -->
#' 
#' <!--
#' ToDo:
#' 
#' How to compute the mean squared error:
#' https://stats.stackexchange.com/questions/107643/how-to-get-the-value-of-mean-squared-error-in-a-linear-regression-in-r
#' 
#' http://r-statistics.co/Linear-Regression.html
#' 
#' Colinearity
#' https://journal.r-project.org/archive/2017/RJ-2017-048/RJ-2017-048.pdf
#' -->

