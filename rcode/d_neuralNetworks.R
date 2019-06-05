# Jan-Philipp Kolb
# Tue Jun 04 12:25:16 2019

## ------------------------------------------------------------------------
library(tidyverse)
library(GGally)
library(magrittr)
library(readr)
library(dplyr)

## ------------------------------------------------------------------------
url<-'http://archive.ics.uci.edu/ml/machine-learning-databases/'
filen <- "00243/yacht_hydrodynamics.data"
colnam <- c('LongPos_COB', 'Prismatic_Coeff','Len_Disp_Ratio',
            'Beam_Draut_Ratio','Length_Beam_Ratio','Froude_Num',
            'Residuary_Resist')
Yacht_Data <- read_table(file = paste0(url,filen),
                         col_names = colnam) %>%
na.omit()


# Split into test and train sets
set.seed(12345)
library(neuralnet)
Yacht_Data_Train <- sample_frac(tbl = Yacht_Data, 
                                replace = FALSE, size = 0.80)
Yacht_Data_Test <- anti_join(Yacht_Data, Yacht_Data_Train)


## ------------------------------------------------------------------------
set.seed(12321)
Yacht_NN1<-neuralnet(Residuary_Resist~LongPos_COB + 
                  Prismatic_Coeff + Len_Disp_Ratio + 
                    Beam_Draut_Ratio+
                       Length_Beam_Ratio +Froude_Num, 
                     data = Yacht_Data_Train)


## ------------------------------------------------------------------------
plot(Yacht_NN1, rep = 'best')


## ------------------------------------------------------------------------
NN1_Train_SSE <- sum((Yacht_NN1$net.result-
                        Yacht_Data_Train[, 7])^2)/2
paste("SSE: ", round(NN1_Train_SSE, 4))


## ------------------------------------------------------------------------
Test_NN1_Output <- compute(Yacht_NN1, 
                           Yacht_Data_Test[, 1:6])$net.result
(NN1_Test_SSE<-sum((Test_NN1_Output-Yacht_Data_Test[,7])^2)/2)


## ------------------------------------------------------------------------
# 2-Hidden Layers, Layer-1 4-neurons, Layer-2, 1-neuron, 
# logistic activation function
set.seed(12321)
Yacht_NN2 <- neuralnet(Residuary_Resist ~ LongPos_COB + 
              Prismatic_Coeff + Len_Disp_Ratio + 
                Beam_Draut_Ratio + 
                Length_Beam_Ratio + Froude_Num, 
                       data = Yacht_Data_Train, 
                       hidden = c(4, 1), 
                       act.fct = "logistic")


## ------------------------------------------------------------------------
NN2_Train_SSE <- sum((Yacht_NN2$net.result - 
                        Yacht_Data_Train[, 7])^2)/2
## Test Error
Test_NN2_Output<-compute(Yacht_NN2, 
            Yacht_Data_Test[, 1:6])$net.result
NN2_Test_SSE <- sum((Test_NN2_Output - 
                       Yacht_Data_Test[, 7])^2)/2


## ------------------------------------------------------------------------
scale11 <- function(x) {
    (2 * ((x - min(x))/(max(x) - min(x)))) - 1
}
Yacht_Data_Train <- Yacht_Data_Train %>% mutate_all(scale11)
Yacht_Data_Test <- Yacht_Data_Test %>% mutate_all(scale11)

# 2-Hidden Layers, Layer-1 4-neurons, Layer-2, 1-neuron, 
# tanh activation function
set.seed(12321)
Yacht_NN3 <- neuralnet(Residuary_Resist ~ LongPos_COB + 
          Prismatic_Coeff+Len_Disp_Ratio+Beam_Draut_Ratio + 
            Length_Beam_Ratio + Froude_Num, 
                       data = Yacht_Data_Train, 
                       hidden = c(4, 1), 
                       act.fct = "tanh")



## ------------------------------------------------------------------------
## Training Error
NN3_Train_SSE <- sum((Yacht_NN3$net.result - 
                        Yacht_Data_Train[, 7])^2)/2
## Test Error
Test_NN3_Output <- compute(Yacht_NN3, 
                           Yacht_Data_Test[, 1:6])$net.result
NN3_Test_SSE <- sum((Test_NN3_Output - 
                       Yacht_Data_Test[, 7])^2)/2


## ------------------------------------------------------------------------
set.seed(12321)
Yacht_NN4 <- neuralnet(Residuary_Resist ~ LongPos_COB + 
                         Prismatic_Coeff + Len_Disp_Ratio + 
                         Beam_Draut_Ratio + Length_Beam_Ratio + 
                         Froude_Num, 
                       data = Yacht_Data_Train, 
                       act.fct = "tanh")
## Training Error
NN4_Train_SSE <- sum((Yacht_NN4$net.result - 
                        Yacht_Data_Train[, 7])^2)/2



## ------------------------------------------------------------------------
## Test Error
Test_NN4_Output <- compute(Yacht_NN4, 
                           Yacht_Data_Test[, 1:6])$net.result
NN4_Test_SSE<-sum((Test_NN4_Output-Yacht_Data_Test[, 7])^2)/2

# Preparing bar plot of results
RegNNErr<-tibble(Network=rep(c("NN1","NN2","NN3","NN4"),each=2), 
                     DataSet=rep(c("Train", "Test"), time = 4), 
                     SSE = c(NN1_Train_SSE, NN1_Test_SSE, 
                             NN2_Train_SSE, NN2_Test_SSE, 
                             NN3_Train_SSE, NN3_Test_SSE, 
                             NN4_Train_SSE, NN4_Test_SSE))


## ------------------------------------------------------------------------
RegNNErr %>% 
  ggplot(aes(Network, SSE, fill = DataSet)) + 
  geom_col(position = "dodge") + 
  ggtitle("Regression ANN's SSE")


## ------------------------------------------------------------------------
plot(Yacht_NN2, rep = "best")


## ----eval=F--------------------------------------------------------------
## set.seed(12321)
## Yacht_NN2 <- neuralnet(Residuary_Resist ~ LongPos_COB +
##                          Prismatic_Coeff + Len_Disp_Ratio +
##                          Beam_Draut_Ratio + Length_Beam_Ratio +
##                          Froude_Num,data = Yacht_Data_Train,
##                        hidden = c(4, 1),
##                        act.fct = "logistic",rep = 10)


## ----eval=F,echo=F-------------------------------------------------------
## save(Yacht_NN2,file="../data/lm_nn_Yacht_NN2.RData")


## ----echo=F--------------------------------------------------------------
load("../data/lm_nn_Yacht_NN2.RData")


## ------------------------------------------------------------------------
plot(Yacht_NN2, rep = "best")


## ------------------------------------------------------------------------
library(tidyverse)
library(neuralnet)
library(GGally)


## ------------------------------------------------------------------------
url1<-'http://archive.ics.uci.edu/ml/machine-learning-'
url2<-"databases//haberman/haberman.data"
url <- paste0(url1,url2)
Hab_Data<-read_csv(file=url,col_names=c('Age', 'Operation_Year', 
                          'Number_Pos_Nodes','Survival')) 


## ------------------------------------------------------------------------
Hab_Data <- Hab_Data%>%
  na.omit() %>%
  mutate(Survival = ifelse(Survival == 2, 0, 1),
         Survival = factor(Survival))


## ----eval=F--------------------------------------------------------------
## ggpairs(Hab_Data,title="Scatterplot Matrix Haberman's
##         Survival Features")


## ------------------------------------------------------------------------
scale01 <- function(x){
  (x - min(x)) / (max(x) - min(x))
}


## ------------------------------------------------------------------------
Hab_Data <- Hab_Data %>%
  mutate(Age = scale01(Age), 
         Operation_Year = scale01(Operation_Year), 
         Number_Pos_Nodes = scale01(Number_Pos_Nodes), 
         Survival = as.numeric(Survival)-1)


## ------------------------------------------------------------------------
Hab_Data <- Hab_Data %>%
  mutate(Survival = as.integer(Survival) - 1, 
         Survival = ifelse(Survival == 1, TRUE, FALSE))


## ------------------------------------------------------------------------
set.seed(123)
Hab_NN1 <- neuralnet(Survival ~ Age + Operation_Year + 
                       Number_Pos_Nodes, 
                     data = Hab_Data, 
                     linear.output = FALSE, 
                     err.fct = 'ce', 
                     likelihood = TRUE)


## ----fig.height=8,fig.width=14-------------------------------------------
plot(Hab_NN1, rep = 'best')


## ------------------------------------------------------------------------
Hab_NN1_Train_Error <- Hab_NN1$result.matrix[1,1]
paste("CE Error: ", round(Hab_NN1_Train_Error, 3)) 
Hab_NN1_AIC <- Hab_NN1$result.matrix[4,1]
paste("AIC: ", round(Hab_NN1_AIC,3))
Hab_NN2_BIC <- Hab_NN1$result.matrix[5,1]
paste("BIC: ", round(Hab_NN2_BIC, 3))


## ------------------------------------------------------------------------
set.seed(123)
# 2-Hidden Layers, Layer-1 2-neurons, Layer-2, 1-neuron
Hab_NN2 <- neuralnet(Survival ~ Age + Operation_Year + 
                  Number_Pos_Nodes,data = Hab_Data, 
                     linear.output = FALSE,err.fct = 'ce', 
                     likelihood = TRUE, hidden = c(2,1))


## ------------------------------------------------------------------------
set.seed(123)
Hab_NN3<-neuralnet(Survival~Age+Operation_Year+
                     Number_Pos_Nodes, 
  data = Hab_Data,linear.output = FALSE,err.fct = 'ce',
  likelihood = TRUE, hidden = c(2,2))


## ------------------------------------------------------------------------
set.seed(123)
Hab_NN4 <- neuralnet(Survival ~ Age + Operation_Year + 
          Number_Pos_Nodes,data = Hab_Data,
          linear.output = FALSE,err.fct = 'ce',
          likelihood = TRUE,hidden = c(1,2))


## ------------------------------------------------------------------------

Class_NN_ICs<-tibble('Network'=rep(c("NN1","NN2","NN3","NN4"),
    each = 3),'Metric'=rep(c('AIC', 'BIC', 'ce Error * 100'), 
    length.out = 12),'Value' = c(Hab_NN1$result.matrix[4,1], 
    Hab_NN1$result.matrix[5,1],100*Hab_NN1$result.matrix[1,1],
    Hab_NN2$result.matrix[4,1],Hab_NN2$result.matrix[5,1],
    100*Hab_NN2$result.matrix[1,1],Hab_NN3$result.matrix[4,1],
    Hab_NN3$result.matrix[5,1],100*Hab_NN3$result.matrix[1,1],
    Hab_NN4$result.matrix[4,1],Hab_NN4$result.matrix[5,1],
    100*Hab_NN4$result.matrix[1,1]))


## ----fig.height=6,fig.width=14-------------------------------------------
Class_NN_ICs %>%ggplot(aes(Network,Value,fill=Metric)) +
  geom_col(position = 'dodge')+ggtitle("AIC, BIC, and 
  Cross-Entropy Error of the Classification ANNs", 
  "Note: ce Error displayed is 100 times its true value")


## ------------------------------------------------------------------------
set.seed(500)
library(MASS)
data <- Boston


## ------------------------------------------------------------------------
index <- sample(1:nrow(data),round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]
lm.fit <- glm(medv~., data=train)
summary(lm.fit)
pr.lm <- predict(lm.fit,test)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)


## ------------------------------------------------------------------------
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, 
                              scale = maxs - mins))
train_ <- scaled[index,]
test_ <- scaled[-index,]


## ----eval=F--------------------------------------------------------------
## install.packages("neuralnet")


## ------------------------------------------------------------------------
library(neuralnet)
n <- names(train_)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], 
                                      collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)


## ------------------------------------------------------------------------
plot(nn)

