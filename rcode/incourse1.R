# Jan-Phillip Kolb
# 


# install.packages("lme4")

library(lme4)

install.packages("keras")

# to coop overfitting
install.packages("glmnet")

# xgboost

install.packages("xgboost")

install.packages("rpart")

install.packages("gbm")

install.packages("nnet")

?knn

?kmeans

kmeans()

install.packages("tidyverse")

#############################

path1<-"https://raw.githubusercontent.com/"
path2<- "thomaspernet/data_csv_r/master/data/"
dname <- "titanic_csv.csv"
titanic <- read.csv(paste0(path1,path2,dname))

data(Titanic)
head(Titanic)

install.packages("datasets.load")

install.packages("colourpicker")
c("#8B2323", "#7FFFD4")


# lme4::

### Exercise swiss data

# 1)
data(swiss) 
dim(swiss) 
nrow(swiss)
ncol(swiss)

head(swiss,n=10)
tail(swiss)
View(swiss)
str(swiss) 

# install.packages("DT")

DT::datatable(swiss)

####

data(airquality)

(airq <- data.table::data.table(airquality))

airq

rm(airq)

### Solution: random number 

set.seed(10)
(x <- runif(8))


round(exp(diff(log(x))), 1)

clean_titanic <- titanic %>%
  mutate(pclass=factor(pclass,levels = c(1, 2, 3),
                       labels=c('Upper','Middle','Lower')),
         survived = factor(survived,levels = c(0, 1),
                           labels=c('No', 'Yes'))) %>%
  na.omit()

library(dplyr)

tit_wna <- na.omit(titanic)

# mutate(tit_wna,...)

clean_titanic <- mutate(,pclass=factor(pclass,levels = c(1, 2, 3),
                                              labels=c('Upper','Middle','Lower'))))


numerics <- c(1,2,3)
str(numerics)

charvec <- c("hj",7,"iu")
str(charvec)

ab <- as.factor(c(1,2,1,2))
str(ab)
#########################

library(dplyr)
library(tidyr)
stocks <- tibble(
  time = as.Date('2009-01-01') + 0:9,
  X = rnorm(10, 0, 1),
  Y = rnorm(10, 0, 2),
  Z = rnorm(10, 0, 4)
)


head(gather(stocks, "stock", "price", -time))
