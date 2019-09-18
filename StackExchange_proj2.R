#Joel Hensel, Mark Tenzer, Devan Visvalingam, Ada Zhu

#setwd("/Users/apple/Desktop")
library(car)
library(bestglm)
library(caTools)
library(MASS)
library(olsrr)
library(ggplot2)
library(glmnet)


###############################
#QUESTION ONE
###############################

# # # # # # # # # # # # # # # # # # # # # # # # 
# Data preprocessing
# # # # # # # # # # # # # # # # # # # # # # # # 
rawdata <- read.csv("question_1_updated_again.csv")
rawdata <- rawdata[rawdata$PostTypeId == 1, ]
rawdata$BountyAmount[is.na(rawdata$BountyAmount)] <-  0
rawdata$UserCreationDate <-  as.Date(rawdata$UserCreationDate)
rawdata$UserLastAccessDate <-  as.Date(rawdata$UserLastAccessDate)
rawdata$PostCreationDate <-  as.Date(rawdata$PostCreationDate)
rawdata$PostBody <- as.character(rawdata$Body)
rawdata$PostTitle <- as.character(rawdata$Title)

data2 <- data.frame(
  UserCreationDate = as.numeric(rawdata$UserCreationDate),
  PostCreationDate = as.numeric(rawdata$PostCreationDate),
  LastAccess = as.numeric(rawdata$UserLastAccessDate),
  PostScore = 1/sqrt(rawdata$PostScore - min(rawdata$PostScore) + 1),
  PostViewCount = log(rawdata$PostViewCount),
  HasProfileImg = (rawdata$ProfileImageUrl != ""),
  HasAboutMe = (rawdata$AboutMe != ""),
  HasLocation = (rawdata$Location != ""),
  HasWebsite = (rawdata$WebsiteUrl != ""),
  Reputation = log(rawdata$UserReputation),
  UserViews = 1/sqrt(rawdata$UserViews+1),
  HasBounty = (rawdata$BountyAmount > 0),
  TagCount = lengths(gregexpr(pattern = ">", rawdata$Tags)),
  TitleLen = log(unname(sapply(as.character(rawdata$PostTitle), nchar))),
  BodyLen = log(unname(sapply(as.character(rawdata$PostBody), nchar))),
  y = rawdata$AnswerCount > 0
)

nrow(data2) # 24,978 rows

nrow(data2[data2$y == TRUE,]) # 17079 of observations have at least one answer
nrow(data2[data2$y == FALSE,]) # 7899 of observations have no answer

17079 / 24978    ### 68% of the observations get an answer to their question


# # # # # # # # # # # # # # # # # # # # # # # # 
# Model 1.1: Logistic regression: is there any answer at all?
# # # # # # # # # # # # # # # # # # # # # # # # 

# Trainning and Testing ## Model Selection
set.seed(123)
sample2<- sample.split(data2,SplitRatio = 0.8) 
training2 <- subset(data2,sample2 ==TRUE) 
testing2 <- subset(data2, sample2==FALSE)

### Full Model
glm_model <- glm(y ~ ., family = binomial(link = "logit"), data = training2)
summary(glm_model)
# Null Deviance: 23353
# Res Deviance: 19261
#AIC = 19293
#highest pvalues: LastAccess .25, HasProfileImg .93, HasAboutMe .77, HasLocation .09, HasWebsiteTrue .078, TitleLen .15, Reputation, .06

vif(glm_model)
#All Variance inflation factors are under 5, meaning there is little multicollinearity to be concerned with
# UserCreationDate, PostCreationDate, and Reputation had the relative highest VIF values
pred <- predict(glm_model, testing2, type="response")
sum((pred > 0.5) == testing2$y) / nrow(testing2) #.73398 Accuracy Rate for this 1st model


#Although there is little multicollinearity, we wanted to check the stepwise AIC method to lower the current AIC if we can
stepAIC(glm_model)
## Suggested removing HasAboutMe, HasProfileImg, and LastAccess
##New AIC 19290, reduced, but not by a significant amount

### New AIC Reduced model
AIC_model <- glm(formula = y ~ UserCreationDate + PostCreationDate + PostScore + 
                   PostViewCount + HasLocation + HasWebsite + Reputation + UserViews + 
                   HasBounty + TagCount + TitleLen + BodyLen, family = binomial(link = "logit"), 
                 data = training2)
summary(AIC_model)
#Almost all regressors seem to have significance, according to the t-tests of each variable. Exception is TitleLen, p = .15
#The reduced model pruned the follow three regressors from the original model: ‘HasProfileImg’, ‘HasAboutme’, and ‘Last Access’.
vif(AIC_model)
# The VIFs for UserCreationDate, PostCreationDate, and Reputation all went down. All of the VIFS also became a bit smaller

AIC_model_pred <- predict(AIC_model, testing2, type="response")
sum((AIC_model_pred > 0.5) == testing2$y) / nrow(testing2) #.7339846 accuracy
#The predictive accuracy has not gotten any better

final_reduced_model <- glm(formula = y ~ UserCreationDate + PostCreationDate + PostScore +
                             PostViewCount + HasLocation + HasWebsite + Reputation + UserViews +
                             HasBounty + TagCount + TitleLen + BodyLen, family = binomial(link = "logit"),
                           data = data2)
summary(final_reduced_model)


# # # # # # # # # # # # # # # # # # # # # # # # 
# Model 1.2: Linear regression: number of answers
# # # # # # # # # # # # # # # # # # # # # # # # 
rawdata$FavoriteCount[is.na(rawdata$FavoriteCount)] <-  0

data3 <- data.frame(
  UserCreationDate = as.numeric(rawdata$UserCreationDate),
  PostCreationDate = as.numeric(rawdata$PostCreationDate),
  LastAccess = as.numeric(rawdata$UserLastAccessDate),
  PostScore = 1/sqrt(rawdata$PostScore - min(rawdata$PostScore) + 1),
  PostViewCount = log(rawdata$PostViewCount),
  HasProfileImg = (rawdata$ProfileImageUrl != ""),
  HasAboutMe = (rawdata$AboutMe != ""),
  HasLocation = (rawdata$Location != ""),
  HasWebsite = (rawdata$WebsiteUrl != ""),
  Reputation = log(rawdata$UserReputation),
  UserViews = 1/sqrt(rawdata$UserViews+1),
  HasBounty = (rawdata$BountyAmount > 0),
  TagCount = lengths(gregexpr(pattern = ">", rawdata$Tags)),
  TitleLen = log(unname(sapply(as.character(rawdata$PostTitle), nchar))),
  BodyLen = log(unname(sapply(as.character(rawdata$PostBody), nchar))),
  CommentCt = log(rawdata$CommentCount + 1),
  FavCt = log(rawdata$FavoriteCount + 1),
  y  = rawdata$AnswerCount
)
# Do 100 train-test splits (80/20) to estimate out-of-sample performance.
# Compare R^2 between a Poisson GLM and a log-transformed linear regression.
# Although R^2 is usually associated with linear regression, Poisson scores higher!
pred_r2_all <- c()
pred_r2_all_lm <- c()
for (i in 1:100) {
  
  # Split into training/testing
  sample3<- sample.split(data3,SplitRatio = 0.8) 
  trainning3 <- subset(data3,sample3 ==TRUE)
  testing3 <- subset(data3, sample3==FALSE)
  
  # Fit Poisson GLM and log linear regression
  # y is incremented before log, because it may equal 0. log(y+1), however, is always defined
  poisson_lm <- glm(y ~ ., family = poisson, data = trainning3)
  lm <- lm(I(log(y+1)) ~ ., data = trainning3)
  
  # Generate predictions and calculate out-of-sample R^2 for Poisson
  pred3 <- predict(poisson_lm, testing3, type="response")
  pred_r2 <- 1 - (sum((testing3$y - pred3)^2) / sum((testing3$y - mean(testing3$y))^2))
  
  # Generate predictions and calculate out-of-sample R^2 for linear
  pred3 <- predict(lm, testing3)
  logy <- log(testing3$y+1)
  pred_r2_lm <- 1 - (sum((logy - pred3)^2) / sum((logy - mean(logy))^2))
  
  # Store the pred R^2
  pred_r2_all <- c(pred_r2_all, pred_r2)
  pred_r2_all_lm <- c(pred_r2_all_lm, pred_r2_lm)
  
}
# Report the mean over all trials
mean(pred_r2_all) # result for Poisson: 0.3547323
mean(pred_r2_all_lm) # result for linear (log-transformed y): 0.3243562

# Excluding influential points only makes things worse...
# Find influential points
influence <- influence.measures(poisson_lm)
# Exclude influential points based on DFfits.  Using other metrics yields similar results
trainning3_noinf <- trainning3[influence$is.inf[, "dffit"] == 0, ]
# Refit Poisson regression
poisson_lm_noinf <- glm(y ~ ., family = poisson, data = trainning3_noinf)
# Generate predictions and calculate R^2
pred3_noinf <- predict(poisson_lm_noinf, testing3, type="response")
1 - (sum((testing3$y - pred3_noinf)^2) / sum((testing3$y - mean(testing3$y))^2))
# result excluding influential points: 0.3449689
# compare to:
pred_r2 # result including all points: 0.3548327

# anova(glm(y ~ 1, family = poisson, data = data3), poisson_lm)

# Use cross-validation to find optimal lambda for ridge regression
# Possible lambda values
possible_k <- exp(seq(-6, -2, .05))
# Rewrite training data as a model matrix
X <- model.matrix( ~ . - y, data = trainning3)
# Ridge regression cross-validation
mod_ridge <- cv.glmnet(x = X,
                       y = as.matrix(trainning3$y),
                       family = "poisson",
                       alpha = "0", 
                       standardize = T, # default is T 
                       intercept = T,
                       nfolds = 10,
                       lambda = possible_k)
plot(mod_ridge)
mod_ridge$lambda.min # optimal lambda for ridge regression: 0.01005184

# Use cross-validation to find optimal lambda for lasso regression
mod_lasso <- cv.glmnet(x = X,
                       y = as.matrix(trainning3$y),
                       family = "poisson",
                       alpha = "1", 
                       standardize = T, # default is T 
                       intercept = T,
                       nfolds = 10,
                       lambda = possible_k)
plot(mod_lasso)
mod_lasso$lambda.min # optimal lambda for lasso regression: 0.004086771

# The lambda values seem too small to make a difference. But try anyway for ridge, since it is larger, to demonstrate
# Fit Poisson regression with optimal ridge penalty, to the training data
mod_ridge_final <- glmnet(x = X, 
                          y = as.matrix(trainning3$y),
                          family = "poisson",
                          alpha = "0",
                          standardize = T, # default is T 
                          intercept = T,
                          lambda = mod_ridge$lambda.min)
# Reformat testing data as a model matrix
Xtest <- model.matrix( ~ . - y, data = testing3)
# Generate predictions
pred3RR <- predict(mod_ridge_final, Xtest, type="response")
# This produces essentially the same result.
pred_r2_rr <- 1 - (sum((testing3$y - pred3RR)^2) / sum((testing3$y - mean(testing3$y))^2))
# result: 0.3556198

# Check VIFs of current model (no regularization; Poisson) before proceeding to model selection
poisson_semifinal <- glm(y ~ ., family = poisson, data = data3)
vif(poisson_semifinal)
#UserCreationDate PostCreationDate       LastAccess        PostScore    PostViewCount    HasProfileImg       HasAboutMe 
#5.692462         5.099716         1.710036         5.527573         2.701461         1.539076         1.638452 
#HasLocation       HasWebsite       Reputation        UserViews        HasBounty         TagCount         TitleLen 
#1.612582         1.471502         3.561145         2.634564         1.095743         1.094185         1.032115 
#BodyLen        CommentCt            FavCt 
#1.102779         1.045007         4.826000

# Run 100 model selections on the training data, and validate using out-of-sample (testing) pred R^2.
# This barely changes the mean result--by less than 0.01.
pred_r2_aic_all <- c()
formulae_AIC <- c()
for (i in 1:100) {
  
  print(i)
  
  # Split data into training/testing
  sample3<- sample.split(data3,SplitRatio = 0.8) 
  trainning3 <- subset(data3,sample3 ==TRUE)
  testing3 <- subset(data3, sample3==FALSE)
  
  # Fit Poisson regression full model
  poisson_lm <- glm(y ~ ., family = poisson, data = trainning3)
  
  # Do stepwise regression, using AIC as criterion
  # There are too many predictors to do exhaustive search 100 times!
  best <- stepAIC(poisson_lm, direction = "both", trace=FALSE)
  # Fit model with the best AIC
  poisson_AIC <- glm(best$formula, family = poisson, data = data3)
  
  # Generate predictions and calculate pred R^2
  pred3_AIC <- predict(poisson_AIC, testing3, type="response")
  pred_r2_AIC<- 1 - (sum((testing3$y - pred3_AIC)^2) / sum((testing3$y - mean(testing3$y))^2))
  
  # Store R^2 and the best-AIC formula object
  pred_r2_aic_all <- c(pred_r2_aic_all, pred_r2_AIC)
  formulae_AIC <- c(formulae_AIC, best$formula)
  
}
# Mean pred R^2
mean(pred_r2_aic_all)
# result: 0.3522462

# Though the resulting R^2 barely changes when using model selection,
# there are fewer predictors (the model is simpler)
# and the maximum VIF decreases! So the results are better.
best <- stepAIC(poisson_semifinal, direction = "both", trace=TRUE)
poisson_AIC <- glm(best$formula, family = poisson, data = data3)
vif(poisson_AIC )
#UserCreationDate PostCreationDate        PostScore    PostViewCount    HasProfileImg      HasLocation       HasWebsite 
#5.158642         4.566489         5.119727         2.640303         1.507849         1.344048         1.381062 
#UserViews        HasBounty         TagCount          BodyLen        CommentCt            FavCt 
#1.647417         1.095091         1.082907         1.089004         1.043483         4.795518

# Summary of final results
summary(poisson_AIC)
# All coefficients are significant (Wald test: p < 0.05) or near-significant (Wald test: p < 0.1)

# Confidence intervals for betas
confint(poisson_AIC)
# Note that transformations on PostScore and UserViews
# flip the sign--increasing the score of the post or user's views DECREASES the transformed variable.
# When interpreting sign of these 2 coefficients, flip the sign.
#                           2.5 %        97.5 %
#   (Intercept)      7.768872e-01  1.611349e+00
# UserCreationDate   1.531194e-05  8.070741e-05
# PostCreationDate  -7.855841e-05 -1.694185e-05
# PostScore         -5.628091e+00 -4.489318e+00
# PostViewCount      1.631571e-01  1.825649e-01
# HasProfileImgTRUE -6.105260e-02  7.111067e-04
# HasLocationTRUE    1.675562e-02  7.903884e-02
# HasWebsiteTRUE    -5.787990e-03  7.355726e-02
# UserViews         -1.678660e-01 -5.194048e-02
# HasBountyTRUE      1.583966e-01  2.776964e-01
# TagCount          -5.706621e-02 -3.565116e-02
# BodyLen           -1.043392e-01 -7.241349e-02
# CommentCt         -5.448868e-02 -2.213055e-02
# FavCt              3.269478e-03  5.557706e-02

v = vif(poisson_semifinal)
v2 = vif(poisson_AIC)
ggplot() + geom_col(aes(x=names(v), y=v, fill = (names(v) %in% names(v2)), alpha=0.5))+ geom_col(aes(x=names(v2), y=v2), alpha=0.5) + xlab("Predictor") + ylab("VIF")



# # # # # # # # # # # # # # # # # # # # # # # # 
# Model 1.3: Is There an Accepted Answer?
# # # # # # # # # # # # # # # # # # # # # # # # 

data <- data.frame(
  UserCreationDate = as.numeric(rawdata$UserCreationDate),
  PostCreationDate = as.numeric(rawdata$PostCreationDate),
  LastAccess = as.numeric(rawdata$UserLastAccessDate),
  PostScore = 1/sqrt(rawdata$PostScore - min(rawdata$PostScore) + 1),
  PostViewCount = log(rawdata$PostViewCount),
  HasProfileImg = (rawdata$ProfileImageUrl != ""),
  HasAboutMe = (rawdata$AboutMe != ""),
  HasLocation = (rawdata$Location != ""),
  HasWebsite = (rawdata$WebsiteUrl != ""),
  Reputation = log(rawdata$UserReputation),
  UserViews = 1/sqrt(rawdata$UserViews+1),
  HasBounty = (rawdata$BountyAmount > 0),
  TagCount = lengths(gregexpr(pattern = ">", rawdata$Tags)),
  TitleLen = log(unname(sapply(as.character(rawdata$PostTitle), nchar))),
  BodyLen = log(unname(sapply(as.character(rawdata$PostBody), nchar))),
  CommentCount = log(rawdata$CommentCount + 1),
  FavoriteCount = log(rawdata$FavoriteCount + 1),
  y  = !is.na(rawdata$AcceptedAnswerId) #TRUE /FALSE
)

# Split of dataset into training dataset and testing dataset
set.seed(123)  
sample<- sample.split(data,SplitRatio = 0.8) 
training <- subset(data,sample ==TRUE) 
testing <- subset(data, sample==FALSE)

# Model selection
#This is logistic regression, as the response variable (whether there is an accepted answer or not) is binary
acceptedanswermodel <- glm(y ~ ., family = binomial(link = "logit"), data = training)
summary(acceptedanswermodel)

# AIC Analysis
stepAIC(acceptedanswermodel)
#Suggests elminiating TitleLen and HasWebsite

#Likelihood Ratio Tests
#This model has 17 regressors and an intercept. We will try to prune the number of regressors 
#First we remove "HasAboutMe" and "HasWebsite"
#Null hypothesis: reduced model is true
reducedmodel <- glm(y ~ UserCreationDate + PostCreationDate + LastAccess + PostScore + PostViewCount + HasProfileImg + HasLocation + Reputation + UserViews + HasBounty + TagCount + TitleLen + BodyLen  + CommentCount + FavoriteCount, family = binomial(link = "logit"), data = training)
anova(reducedmodel, acceptedanswermodel)
anovadeviance <- anova(reducedmodel, acceptedanswermodel)$Deviance[2] #3.67
anovadf <- anova(reducedmodel, acceptedanswermodel)$Df[2] #2
1 - pchisq(anovadeviance, anovadf) #0.16
#0.16 > 0.05, so fail to reject and keep the reduced model
acceptedanswermodel = reducedmodel
summary(acceptedanswermodel)
#This model has 15 regressors and an intercept. We will try to prune the number of regressors 
#We remove "UserViews" and "TitleLen"
#Null hypothesis: reduced model is true
reducedmodel <- glm(y ~ UserCreationDate + PostCreationDate + LastAccess + PostScore + PostViewCount + HasProfileImg + HasLocation + Reputation + HasBounty + TagCount + BodyLen + CommentCount + FavoriteCount, family = binomial(link = "logit"), data = training)
anova(reducedmodel, acceptedanswermodel)
anovadeviance <- anova(reducedmodel, acceptedanswermodel)$Deviance[2] #3.9
anovadf <- anova(reducedmodel, acceptedanswermodel)$Df[2] #2
1 - pchisq(anovadeviance, anovadf) #0.14
#0.14 > 0.05, so fail to reject and keep the reduced model
acceptedanswermodel = reducedmodel
summary(acceptedanswermodel)
#This model has 13 regressors and an intercept. We will try to prune the number of regressors 
#We remove "HasBounty"
#Null hypothesis: reduced model is true
reducedmodel <- glm(y ~ UserCreationDate + PostCreationDate + LastAccess + PostScore + PostViewCount + HasProfileImg + HasLocation + Reputation + TagCount + BodyLen + CommentCount + FavoriteCount, family = binomial(link = "logit"), data = training)
anova(reducedmodel, acceptedanswermodel)
anovadeviance <- anova(reducedmodel, acceptedanswermodel)$Deviance[2] #12.577
anovadf <- anova(reducedmodel, acceptedanswermodel)$Df[2] #1
1 - pchisq(anovadeviance, anovadf) #0.002
#0.002 < 0.05, so reject and keep our model
summary(acceptedanswermodel)

#Let's compare this to the null model
acceptedanswernullmodel <- glm(formula = y ~ 1, family = binomial(link = "logit"), data = training)
anova(acceptedanswernullmodel, acceptedanswermodel)

anova(acceptedanswernullmodel, acceptedanswermodel)$'Deviance'[2]
anova(acceptedanswernullmodel, acceptedanswermodel)$'Df'[2]
1 - pchisq(anova(acceptedanswernullmodel, acceptedanswermodel)$'Deviance'[2],anova(acceptedanswernullmodel, acceptedanswermodel)$'Df'[2])
#0 --> reject and keep our model

#Ultimately for our model, we choose to take the model with the lowest AIC, but then remove 1 more regressor. The AIC metric increases by less than a thousandth of a percent, but it does simplify the model.
acceptedanswermodel <- glm(y ~ UserCreationDate + PostCreationDate + LastAccess + PostScore + PostViewCount + HasProfileImg + HasLocation + Reputation + UserViews + HasBounty + TagCount + BodyLen  + CommentCount + FavoriteCount, family = binomial(link = "logit"), data = training)
summary(acceptedanswermodel)

# VIFs
vif(acceptedanswermodel)
#The maximum VIF is 4.5 < 5, so no concerns raised here

# Cross-validation
acceptedanswerpreds <- predict(acceptedanswermodel, testing, type="response")
sum((acceptedanswerpreds > 0.5) == testing$y) / nrow(testing) #70.2% Accuracy Rate.

#Fitting model on the entire dataset
acceptedanswermodel <- glm(y ~ UserCreationDate + PostCreationDate + LastAccess + PostScore + PostViewCount + HasProfileImg + HasLocation + Reputation + UserViews + HasBounty + TagCount + BodyLen  + CommentCount + FavoriteCount, family = binomial(link = "logit"), data = data)
summary(acceptedanswermodel)




###############################
#QUESTION TWO
###############################

# # # # # # # # # # # # # # # # # # # # # # # # 
# Model 2: Linear regression: scores of answers
# # # # # # # # # # # # # # # # # # # # # # # # 
rawdata <- read.csv("Answerer_Query.csv")#different data than the one from above

#convert into Date variables
rawdata$answer_lastactivitydate <-  as.Date(rawdata$answer_lastactivitydate)
rawdata$post_creationdate <-  as.Date(rawdata$post_creationdate)
rawdata$user_creationdate <-  as.Date(rawdata$user_creationdate)
rawdata <- rawdata[, -5] #remove viewCount cuz its empty in dataset

#check variable distributions that might be skewed
hist(rawdata$answer_score)
summary(rawdata$answer_score)
#    Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#-27.000    0.000    1.000    6.779    3.000 6907.000 

hist(rawdata$Answer_Length)
summary(rawdata$Answer_Length)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#19.0   286.0   507.0   751.9   896.0 26291.0 
hist(rawdata$question_viewcount)
hist(rawdata$User_Reputation)
hist(rawdata$User_Views)


data <- data.frame(
  score = log(rawdata$answer_score - min(rawdata$answer_score) + 1), #ensure all scores are positive before taking the log
  length = log(rawdata$Answer_Length),
  duration = as.numeric(rawdata$answer_lastactivitydate - rawdata$post_creationdate) +1,
  questionView = log((rawdata$question_viewcount)),
  TitleLen = (unname(sapply(as.character(rawdata$question_title), nchar))),
  HasProfileImg = as.factor(rawdata$ProfImageListed),
  HasAboutMe = as.factor(rawdata$AboutMe),
  Reputation = log(rawdata$User_Reputation),
  UserViews = log(rawdata$User_Views+1),
  TagCount = lengths(gregexpr(">", rawdata$Question_Tags)) #checking regular expression
)


#double check distributions of var
hist(data$score)
hist(data$length)
hist(data$questionView)

#split up testing and training
sample4<- sample.split(data,SplitRatio = 0.8) 
training4 <- subset(data,sample4 ==TRUE)
testing4 <- subset(data, sample4==FALSE)

#linear regression, predicting the score of each answer
mod1 <- lm(score ~ ., data = training4)
summary(mod1) # p-value: < 2.2e-16

#boxcox gives similar result
# bc <- boxcox(mod1)
# lambda <- bc$x[which.max(bc$y)]
# training4$score <- training4$score ^ lambda 
# mod_boxcox <- lm(score ~ ., data = training4)
# summary(mod_boxcox)

# plot cook's distance
cooksd <- cooks.distance(mod1) 
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")

mod_train <- stepAIC(mod1, direction = "both", trace = FALSE) #removed HasProfileImg
summary(mod_train)


#multicollinearity
vif(mod_train) #small vif for all coefficients, implying that theres no multicollinearity

#check the 95% confident interval for coefficients
confint(mod_train)

#check on testing
pred4 <- predict(mod_train, testing4)
mean((pred4 - testing4$score)^2) #[1] 0.0989831 mse
1 - (sum((testing4$score - pred4)^2) / sum((testing4$score - mean(testing4$score))^2)) #0.3365966 r^2


#check for residual assumptions
#normality
extern_s_resids <- studres(mod_train)
hist(extern_s_resids)
qqnorm(extern_s_resids)
qqline(extern_s_resids)

#no obvious trend
plot(fitted.values(mod_train), extern_s_resids)

#no obvious trend
plot(training4$length, extern_s_resids)
plot(training4$duration, extern_s_resids)
plot(training4$questionView, extern_s_resids)
plot(training4$TitleLen, extern_s_resids)
plot(training4$HasAboutMe, extern_s_resids)
plot(training4$Reputation, extern_s_resids)
plot(training4$UserViews, extern_s_resids)
plot(training4$TagCount, extern_s_resids)

#build the final model on all data
mod_2_final <-lm(formula = score ~ length + duration+questionView +TitleLen +HasAboutMe +Reputation +
                 UserViews + TagCount, data = data)
summary(mod_2_final) #0.318, p-value: < 2.2e-16
