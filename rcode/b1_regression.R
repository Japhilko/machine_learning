#' ---
#' title: "Machine Learning: Regression in R"
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
## ----setupregression, include=FALSE--------------------------------------
knitr::opts_chunk$set(echo = T,warning = F,message = F)
pres=T

#' 
#' ## Why a part on simple regression
#' 
#' - OLS can be seen as a simple machine learning technique
#' - Some other machine learning concepts are based on regression (e.g. regularization).
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
#' ### Take all available predictors
#' 
## ------------------------------------------------------------------------
m3_a<-lm(mpg~.,data=mtcars) 

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
## summary(m3b)
## summary(m3c)

#' 
#' 
#' ### Take the logarithm
#' 
## ------------------------------------------------------------------------
m3d<-lm(mpg~log(wt),data=mtcars) 

#' 
#' <!--
#' https://www.r-bloggers.com/r-tutorial-series-regression-with-interaction-variables/
#' 
#' https://www.r-bloggers.com/interpreting-interaction-coefficient-in-r-part1-lm/
#' -->
#' 
#' ## The command `setdiff`
#' 
#' - We can use the command to create a dataset with only the features, without the dependent variable
#' 
## ------------------------------------------------------------------------
names(mtcars)
features <- setdiff(names(mtcars), "mpg")
features

#' 
## ------------------------------------------------------------------------
featdat <- mtcars[,features]

#' 
#' 
#' 
#' ## The command `model.matrix`
#' 
#' <!--
#' - Construct Design Matrices
#' 
#' https://genomicsclass.github.io/book/pages/expressing_design_formula.html
#' -->
#' 
#' - With `model.matrix` the qualitative variables are automatically dummy encoded 
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
#' ## Another example for object orientation
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
#' ### Visualizing residuals
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
#' ## The mean squared error (mse)
#' 
#' - The [**MSE**](https://en.wikipedia.org/wiki/Mean_squared_error) measures the average of the squares of the errors
#' - [**The lower the better**](http://r-statistics.co/Linear-Regression.html)
#' 
## ------------------------------------------------------------------------
(mse5 <- mean((mtcars$mpg -  pre)^2)) # model 5
(mse3 <- mean((mtcars$mpg -  predict(m3))^2)) 

#' 
#' <!--
#' https://stats.stackexchange.com/questions/107643/how-to-get-the-value-of-mean-squared-error-in-a-linear-regression-in-r
#' -->
#' 
#' ### Package `Metrics` to compute mse
#' 
## ----eval=F,echo=F-------------------------------------------------------
## install.packages("Metrics")

#' 
## ------------------------------------------------------------------------
library(Metrics)
mse(mtcars$mpg,predict(m3))

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
#' ## [The bias-variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) (I)
#' 
#' - The bias–variance tradeoff is the property of a set of predictive models whereby models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa. 
#' 
#' [![](figure/bias_variance_tradeoff2.png)](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
#' 
#' <!--
#' https://lbelzile.github.io/lineaRmodels/bias-and-variance-tradeoff.html
#' http://www.sthda.com/english/articles/38-regression-model-validation/157-cross-validation-essentials-in-r/
#' https://daviddalpiaz.github.io/r4sl/biasvariance-tradeoff.html
#' -->
#' 
#' ## The bias-variance tradeoff (II)
#' 
#' ![](figure/bias_variance_tradeoff.PNG)
#' 
#' ## Exercise: regression Ames housing data
#' 
#' 1) Install the package `AmesHousing` and create a [**processed version**](https://cran.r-project.org/web/packages/AmesHousing/AmesHousing.pdf) of the Ames housing data with (at least) the variables `Sale_Price`, `Gr_Liv_Area` and `TotRms_AbvGrd`
#' 2) Create a regression model with `Sale_Price` as dependent and `Gr_Liv_Area` and `TotRms_AbvGrd` as independent variables. Then create seperated models for the two independent variables. Compare the results. What do you think?
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
#' ## A correlation plot
#' 
#' <!--
#' https://www.r-bloggers.com/variable-importance-plot-and-variable-selection/
#' -->
#' 
## ----eval=F,echo=F-------------------------------------------------------
## install.packages("corrplot")

#' 
#' 
## ------------------------------------------------------------------------
library(corrplot)
corrplot(cor(ames_data[,c("Sale_Price","Gr_Liv_Area","TotRms_AbvGrd")]))

#' 
#' 
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
#' - [**Cross Validation **](http://www.sthda.com/english/articles/38-regression-model-validation/157-cross-validation-essentials-in-r/)
#' - Train with more data
#' - Remove features
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
#' <!--
#' ### Swiss Fertility and Socioeconomic Indicators
#' 
## ------------------------------------------------------------------------
data("swiss")

#' -->
#' 
#' ## [Cross Validation in R](http://www.sthda.com/english/articles/38-regression-model-validation/157-cross-validation-essentials-in-r/)
#' 
#' 
#' ### Split data into training and testing dataset
#' 
## ----eval=F,echo=F-------------------------------------------------------
## training.samples <- swiss$Fertility %>%
## createDataPartition(p = 0.8, list = FALSE)
## train.data  <- swiss[training.samples, ]
## test.data <- swiss[-training.samples, ]

#' 
## ------------------------------------------------------------------------
training.samples <- ames_data$Sale_Price %>%
createDataPartition(p = 0.8, list = FALSE)
train.data  <- ames_data[training.samples, ]
test.data <- ames_data[-training.samples, ]

#' 
#' 
#' ### Build the model and make predictions
#' 
#' <!--
#' # Make predictions and compute the R2, RMSE and MAE
#' -->
#' 
## ----eval=F,echo=F-------------------------------------------------------
## model <- lm(Fertility ~., data = train.data)
## (predictions <- model %>% predict(test.data))

#' 
## ------------------------------------------------------------------------
model <- lm(Sale_Price ~ Gr_Liv_Area + TotRms_AbvGrd, 
            data = train.data)
# Make predictions and compute the R2, RMSE and MAE
(predictions <- model %>% predict(test.data))

#' 
#' 
#' ## Model with cross validation
#' 
#' - Loocv: [**leave one out cross validation**](https://machinelearningmastery.com/how-to-estimate-model-accuracy-in-r-using-the-caret-package/)
#' 
## ------------------------------------------------------------------------
train.control <- caret::trainControl(method = "LOOCV")

#' 
## ----eval=F,echo=F-------------------------------------------------------
## # Train the model
## model2 <- train(Fertility ~., data = swiss, method = "lm",
##                trControl = train.control)
## model2 %>% predict(test.data)

#' 
## ----eval=F--------------------------------------------------------------
## # Train the model
## model2 <- train(Sale_Price ~ Gr_Liv_Area + TotRms_AbvGrd,
##                data = train.data, method = "lm",
##                trControl = train.control)
## model2 %>% predict(test.data)

#' 
## ----eval=F,echo=F-------------------------------------------------------
## save(model2,file="../data/ml_ols_cv_model2.RData")

#' 
## ----echo=F--------------------------------------------------------------
load("../data/ml_ols_cv_model2.RData")

#' 
#' 
#' <!--
#' ## [k-fold cross validation](https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/)
#' -->
#' 
#' ## Summarize the results
#' 
## ------------------------------------------------------------------------
summary(model2)$coefficients

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
#' 
#' <!--
#' http://r-statistics.co/Linear-Regression.html
#' https://machinelearningmastery.com/linear-regression-in-r/
#' https://journal.r-project.org/archive/2017/RJ-2017-048/RJ-2017-048.pdf
#' https://cran.r-project.org/web/packages/Metrics/Metrics.pdf
#' -->
#' 
#' 
#' <!--
#' ToDo Liste
#' 
#' Den Effekt von cross validation zeigen
#' -->
