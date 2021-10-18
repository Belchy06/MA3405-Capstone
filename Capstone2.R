library(ggplot2)
library(ggfortify)
library(dplyr)
library(class)
library(caret)
library(e1071)
library(ROCR)
library(tidyverse)
library(boot)
library(rpart)
library(rpart.plot)
library(randomForest)
library(bestglm)
require(caTools)
library(AMR)
library(ggfortify)
install.packages("devtools", repo="http://cran.us.r-project.org")
library(devtools)
install_github("vqv/ggbiplot")
set.seed(1)
library(ggbiplot)
rm(list=ls())
importedData <- read.csv(file = "csv_result-Autism-Adult-Data.csv")

###############################
# Cleaning
###############################
# The data has some spelling errors in the column names, fix these
names(importedData)[names(importedData) == "contry_of_res"] <- "country_of_res"
names(importedData)[names(importedData) == "austim"] <- "autism"

# Remove rows with empty values
cleanData <- importedData[apply(importedData, 1, function(row) all(row != '?')), ]
# Remove age outlier
cleanData <- cleanData[!(cleanData$id %in% c(53)), ]
# Remove unnecessary columns
cleanData <- cleanData[ , !(names(cleanData) %in% c('id',  'used_app_before', 'age_desc', 'relation', 'result'))]
# Age is a char, let's fix this
cleanData$age <- as.numeric(cleanData$age)
# Outcome is a char, change to a numeric 1 or 0 with 1 being yes
cleanData$Class.ASD <- as.integer(as.factor(cleanData$Class.ASD)) - 1

# Ethnicity has many categories. collapse these to mitigate the curse of dimensionality
fct_count(cleanData$ethnicity)
cleanData$ethnicity <- fct_collapse(cleanData$ethnicity,
                                    White = c("White-European"),
                                    Black = c("Black", "Turkish", "Middle Eastern "),
                                    Asian = c("Asian", "South Asian"),
                                    Other = c("others", "Others", "Pasifika", "Hispanic", "Latino"),
)
fct_count(cleanData$ethnicity)


###############################
# PCA
###############################
pca.data <- cleanData[ , 1:10]
for(i in 1:10) {
  pca.data[, i] <- as.integer(pca.data[, i])
}
pca.out <- prcomp(pca.data, scale = TRUE)
# plot is flipped, so flip it back
# pca.out$rotation <- -pca.out$rotation[, c(1,3)]
# pca.out$x <- -pca.out$x[, c(1,3)]
pca.out$rotation <- -pca.out$rotation
pca.out$x <- -pca.out$x
biplot(pca.out, scale = 0)
# variance explained by each PC
pca.var <- pca.out$sdev^2
# proportion of variance explained
pve <- pca.var / sum(pca.var)
# plot PVE
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")

ggplot_pca(pca.out, groups = cleanData$Class.ASD)
dev.new()
autoplot(pca.out, loadings = TRUE, loadings.label = TRUE)
ggbiplot(pca.out, groups = cleanData$Class.ASD, ellipse = TRUE, circle = TRUE)
###############################
# Possible methods: Logistic Regression, Naive Bayes, Classification Tree, Random Forest
###############################
n <- nrow(cleanData) -1
index <- sample(1:n, n*0.8, replace=FALSE)
trainDat <- cbind(pca.out$x[, 1:3], cleanData[, c(11:15, 17)])[index, ]
testDat <- cbind(pca.out$x[, 1:3], cleanData[, c(11:15, 17)])[-index, ]

# Cross fold validation
nFolds <- 5
folds <- createFolds(trainDat$Class.ASD, k = nFolds)


###############################
# Logistic Regression
###############################
train.acc <- c()
test.acc <- c()
train.acc.temp <- c()
test.acc.temp <- c()
for(i in 1:nFolds) {
  LR.model <- glm(Class.ASD ~ ., family = binomial("logit"), trainDat[-folds[[i]], ], control = list(maxit = 50))
  trainPred <- predict(LR.model, newdata = trainDat[-folds[[i]], ], type = "response")
  LR.pred <- rep(0, dim(trainDat[-folds[[i]], ])[1])
  LR.pred[trainPred > .5] <- 1
  trainTable <- table(LR.pred, trainDat[-folds[[i]], ]$Class.ASD)
  
  
  testPred <- predict(LR.model, newdata = trainDat[folds[[i]], ], type = "response")
  LR.pred <- rep(0, dim(trainDat[folds[[i]], ])[1])
  LR.pred[testPred > .5] <- 1
  testTable <- table(LR.pred, trainDat[folds[[i]], ]$Class.ASD)
  
  train.acc.temp <- c(train.acc.temp, (trainTable[1,1]+trainTable[2,2])/sum(trainTable))
  test.acc.temp <- c(test.acc.temp, (testTable[1,1]+testTable[2,2])/sum(testTable))
}
train.acc <- rbind(train.acc, train.acc.temp)
test.acc <- rbind(test.acc, test.acc.temp)
rowMeans(train.acc)
rowMeans(test.acc)

trainDat$Class.ASD <- as.factor(trainDat$Class.ASD)
LR.model <- glm(Class.ASD ~ ., family = binomial("logit"), trainDat, control = list(maxit = 50))
LR.prob <- predict(LR.model, newdata = testDat, type = "response")
LR.pred <- rep(0, dim(testDat)[1])
LR.pred[LR.prob > .5] <- 1
LR.test.table <- table(LR.pred, testDat$Class.ASD)
LR.test.acc <- (LR.test.table[1,1]+LR.test.table[2,2])/sum(LR.test.table)
LR.test.spec <- LR.test.table[1,1]/(LR.test.table[1,1] + LR.test.table[2,2])
LR.test.sens <- LR.test.table[1,1]/(LR.test.table[1,1] + LR.test.table[2,1])
LR.test.acc
LR.test.spec
LR.test.sens

LR.model <- glm(Class.ASD ~ PC1 + PC3, family = binomial("logit"), trainDat, control = list(maxit = 50))
LR.prob <- predict(LR.model, newdata = testDat, type = "response")
LR.pred <- rep(0, dim(testDat)[1])
LR.pred[LR.prob > .5] <- 1
LR.test.table <- table(LR.pred, testDat$Class.ASD)
LR.test.acc <- (LR.test.table[1,1]+LR.test.table[2,2])/sum(LR.test.table)
LR.test.spec <- LR.test.table[1,1]/(LR.test.table[1,1] + LR.test.table[2,2])
LR.test.sens <- LR.test.table[1,1]/(LR.test.table[1,1] + LR.test.table[2,1])
LR.test.acc
LR.test.spec
LR.test.sens



tempData <- trainDat
tempData$gender <- as.factor(trainDat$gender)
tempData$jundice <- as.factor(trainDat$jundice)
tempData$autism <- as.factor(trainDat$autism)
best.LR <- bestglm(tempData, family = binomial(), IC="CV", t=5)
best.LR

###############################
# Naive Bayes
###############################
train.acc <- c()
test.acc <- c()
train.acc.temp <- c()
test.acc.temp <- c()
for(i in 1:nFolds) {
  NB.model <- naiveBayes(Class.ASD ~ ., data=trainDat[-folds[[i]], ])
  trainPred <- predict(NB.model, newdata = trainDat[-folds[[i]], ], type = "class")
  trainTable <- table(trainPred, trainDat[-folds[[i]], ]$Class.ASD)
  
  
  testPred <- predict(NB.model, newdata = trainDat[folds[[i]], ], type = "class")
  testTable <- table(testPred, trainDat[folds[[i]], ]$Class.ASD)
  
  train.acc.temp <- c(train.acc.temp, (trainTable[1,1]+trainTable[2,2])/sum(trainTable))
  test.acc.temp <- c(test.acc.temp, (testTable[1,1]+testTable[2,2])/sum(testTable))
}
train.acc <- rbind(train.acc, train.acc.temp)
test.acc <- rbind(test.acc, test.acc.temp)
rowMeans(train.acc)
rowMeans(test.acc)

NB.model <- naiveBayes(Class.ASD ~ ., data=trainDat)
NB.pred <- predict(NB.model, newdata = testDat, type = "class")
NB.test.table <- table(NB.pred, testDat$Class.ASD)
NB.test.acc <- (NB.test.table[1,1]+NB.test.table[2,2])/sum(NB.test.table)
NB.test.spec <- NB.test.table[1,1]/(NB.test.table[1,1] + NB.test.table[2,2])
NB.test.sens <- NB.test.table[1,1]/(NB.test.table[1,1] + NB.test.table[2,1])
NB.test.acc
NB.test.spec
NB.test.sens



###############################
# Classification Tree
###############################
train.acc <- c()
test.acc <- c()
train.acc.temp <- c()
test.acc.temp <- c()
for(i in 1:nFolds) {
  tree.model <- naiveBayes(Class.ASD ~ ., data=trainDat[-folds[[i]], ])
  trainPred <- predict(tree.model, newdata = trainDat[-folds[[i]], ], type = "class")
  trainTable <- table(trainPred, trainDat[-folds[[i]], ]$Class.ASD)
  
  
  testPred <- predict(tree.model, newdata = trainDat[folds[[i]], ], type = "class")
  testTable <- table(testPred, trainDat[folds[[i]], ]$Class.ASD)
  
  train.acc.temp <- c(train.acc.temp, (trainTable[1,1]+trainTable[2,2])/sum(trainTable))
  test.acc.temp <- c(test.acc.temp, (testTable[1,1]+testTable[2,2])/sum(testTable))
}
train.acc <- rbind(train.acc, train.acc.temp)
test.acc <- rbind(test.acc, test.acc.temp)
rowMeans(train.acc)
rowMeans(test.acc)

tree.model <- rpart(Class.ASD ~ ., data = trainDat, method = "class")
tree.pred <- predict(tree.model, newdata = testDat, type = "class")
tree.test.table <- table(tree.pred, testDat$Class.ASD)
tree.test.acc <- (tree.test.table[1,1]+tree.test.table[2,2])/sum(tree.test.table)
tree.test.spec <- tree.test.table[1,1]/(tree.test.table[1,1] + tree.test.table[2,2])
tree.test.sens <- tree.test.table[1,1]/(tree.test.table[1,1] + tree.test.table[2,1])
tree.test.acc
tree.test.spec
tree.test.sens



###############################
# Random Forest
###############################
train.acc <- c()
test.acc <- c()
train.acc.temp <- c()
test.acc.temp <- c()
for(i in 1:nFolds) {
  RF.model <- naiveBayes(Class.ASD ~ ., data=trainDat[-folds[[i]], ])
  trainPred <- predict(RF.model, newdata = trainDat[-folds[[i]], ], type = "class")
  trainTable <- table(trainPred, trainDat[-folds[[i]], ]$Class.ASD)
  
  
  testPred <- predict(RF.model, newdata = trainDat[folds[[i]], ], type = "class")
  testTable <- table(testPred, trainDat[folds[[i]], ]$Class.ASD)
  
  train.acc.temp <- c(train.acc.temp, (trainTable[1,1]+trainTable[2,2])/sum(trainTable))
  test.acc.temp <- c(test.acc.temp, (testTable[1,1]+testTable[2,2])/sum(testTable))
}
train.acc <- rbind(train.acc, train.acc.temp)
test.acc <- rbind(test.acc, test.acc.temp)
rowMeans(train.acc)
rowMeans(test.acc)

RF.model <- randomForest(Class.ASD ~ ., data=trainDat, method = "class")
RF.pred <- predict(RF.model, newData = testDat, method = "class")
RF.test.table <- table(RF.pred, trainDat$Class.ASD)
RF.test.acc <- (RF.test.table[1,1]+RF.test.table[2,2])/sum(RF.test.table)
RF.test.spec <- RF.test.table[1,1]/(RF.test.table[1,1] + RF.test.table[2,2])
RF.test.sens <- RF.test.table[1,1]/(RF.test.table[1,1] + RF.test.table[2,1])
RF.test.acc
RF.test.spec
RF.test.sens

