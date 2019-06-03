# Jan-Philipp Kolb
# Mon Jun 03 16:47:47 2019
# In course part 2

data(mtcars)

m1 <- lm(mpg~wt,data=mtcars)

sum_mod <- summary(m1)
sum_mod$coefficients

##############################

dev.off()

plot(mtcars$wt,mtcars$mpg)
abline(m1)
segments(mtcars$wt, mtcars$mpg, mtcars$wt, pre, col="red")

#################################


ames_data <- AmesHousing::make_ames()# 1)
# alternative
library(AmesHousing)
ames_data <- make_ames()


colnames(ames_data)
m1 <- lm(Sale_Price ~ Gr_Liv_Area + TotRms_AbvGrd, data = ames_data)
m2 <- lm(Sale_Price ~ Gr_Liv_Area, data = ames_data)
m3 <- lm(Sale_Price ~ TotRms_AbvGrd, data = ames_data)

m1$coefficients
m2$coefficients
m3$coefficients

##########

for (i in 1:3){
  eval(parse(text=paste0("summary(m",i,")")))
}

#################################

?glmnet

library(AmesHousing)
ames_data <- AmesHousing::make_ames()

ncol(ames_data)

ames_train_x <- model.matrix(Sale_Price ~ ., ames_train)[, -1]
ames_train_y <- log(ames_train$Sale_Price)
ames_test_x <- model.matrix(Sale_Price ~ ., ames_test)[, -1]
ames_test_y <- log(ames_test$Sale_Price)

library(glmnet)
ames_ridge <- glmnet(x = ames_train_x,y = ames_train_y,
                     alpha = 0)

coef(ames_ridge)

####################################

install.packages("lars")
library(lars) # 1)
data(diabetes)


library(glmnet) #2)
# Create the scatterplots
set.seed(1234)
par(mfrow=c(2,5))
for(i in 1:10){ # 3)
  plot(diabetes$x[,i], diabetes$y)
  abline(lm(diabetes$y~diabetes$x[,i]),col="red")
}

model_ols <- lm(diabetes$y ~ diabetes$x) # 4)
summary(model_ols)

lambdas <- 10^seq(7, -3)
model_ridge <- glmnet(diabetes$x, diabetes$y, 
                      alpha = 0, lambda = lambdas)
plot.glmnet(model_ridge, xvar = "lambda", label = TRUE)

cv_fit <- cv.glmnet(x=diabetes$x, y=diabetes$y, 
                    alpha = 0, nlambda = 1000)
cv_fit$lambda.min

plot.cv.glmnet(cv_fit)

fit <- glmnet(x=diabetes$x, y=diabetes$y, 
              alpha = 0, lambda=cv_fit$lambda.min)
fit$beta

fit <- glmnet(x=diabetes$x, y=diabetes$y, 
              alpha = 0, lambda=cv_fit$lambda.1se)
fit$beta

# install.packages("rpart")

library(caret)
intrain <- createDataPartition(y=diabetes$y,
                               p = 0.8,
                               list = FALSE)
training <- diabetes[intrain,]
testing <- diabetes[-intrain,]

cv_ridge <- cv.glmnet(x=training$x, y=training$y,
                      alpha = 0, nlambda = 1000)
ridge_reg <- glmnet(x=training$x, y=training$y,
                    alpha = 0, lambda=cv_ridge$lambda.min)
ridge_reg$beta

ridge_reg <- glmnet(x=training$x, y=training$y,
                    alpha = 0, lambda=cv_ridge$lambda.1se)
ridge_reg$beta

ridge_reg <- glmnet(x=training$x, y=training$y,
                    alpha = 0, lambda=cv_ridge$lambda.min)
ridge_pred<-predict.glmnet(ridge_reg,
                           s = cv_ridge$lambda.min,newx = testing$x)
sd((ridge_pred - testing$y)^2)/sqrt(length(testing$y))


ridge_reg <- glmnet(x=training$x, y=training$y,
                    alpha = 0, lambda=cv_ridge$lambda.1se)
ridge_pred <- predict.glmnet(ridge_reg,
                             s = cv_ridge$lambda.1se, newx = testing$x)
sd((ridge_pred - testing$y)^2)/sqrt(length(testing$y))

ols_reg <- lm(y ~ x, data = training)
summary(ols_reg)

ols_pred <- predict(ols_reg, newdata=testing$x,
                    type = "response")
sd((ols_pred - testing$y)^2)/sqrt(length(testing$y))

coef(model_ols)


library(Metrics)
mse(testing$y,ols_pred)
mse(ridge_pred,testing$y)
