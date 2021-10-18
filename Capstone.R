library(ggplot2)
library(ggfortify)
library(dplyr)
library(class)
library(caret)
library(e1071)
library(ROCR)

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
# Remove unnecessary columns
augmentedData <- cleanData[ , !(names(cleanData) %in% c('age_desc'))]
str(augmentedData)

# Age is a char, let's fix this
augmentedData$age <- as.numeric(augmentedData$age)
#check
str(augmentedData)

# Turn the other character columns to factors
augmentedData$gender <- as.integer(as.factor(augmentedData$gender))
augmentedData$ethnicity <- as.integer(as.factor(augmentedData$ethnicity))
augmentedData$jundice <- as.integer(as.factor(augmentedData$jundice))
augmentedData$autism <- as.integer(as.factor(augmentedData$autism))
augmentedData$country_of_res <- as.integer(as.factor(augmentedData$country_of_res))
augmentedData$used_app_before <- as.integer(as.factor(augmentedData$used_app_before))
augmentedData$relation <- as.integer(as.factor(augmentedData$relation))
augmentedData$Class.ASD <- as.integer(as.factor(augmentedData$Class.ASD))
str(augmentedData)
summary(augmentedData)


# 1 = no, 2 = yes

# Age has a maximum value of 383. This is definitely an outlier, should be removed
augmentedData <- augmentedData[!(augmentedData$id %in% c(53)), ]

checked_data <- augmentedData %>%
  rowwise() %>%
  mutate(question_sum = sum(c(A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score))) 
mismatched_data <- checked_data[checked_data$result != checked_data$question_sum, ]


augmentedData <- augmentedData[ , !(names(augmentedData) %in% c('id', 'result'))]

###############################
# PCA
###############################
pca.dat <- augmentedData[ , !(names(augmentedData) %in% c('Class.ASD', 'ethnicity', 'age', 'gender', 'jundice', 'autism', 'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation'))]
for(i in 1:10) {
  pca.dat[ ,i] <- as.integer(pca.dat[ ,i])
}
str(pca.dat)
pr.out <- prcomp(pca.dat, scale = TRUE)
names(pr.out)
# column mean
pr.out$center
# column std dev
pr.out$scale
# principal component loading vector
pr.out$rotation
# plot the first two PCs
biplot(pr.out, scale = 0)

# plot is flipped, so flip it back
pr.out$rotation <- -pr.out$rotation
pr.out$x <- -pr.out$x
dev.new()
biplot(pr.out, scale = 0)
autoplot(pr.out, colour = 'gender', shape = FALSE, label.size = 3)
names(pr.out)
# variance explained by each PC
pr.var <- pr.out$sdev^2
# proportion of variance explained
pve <- pr.var / sum(pr.var)
# plot PVE
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")

table(cleanData$A6_Score, cleanData$A9_Score)

###############################
# Possible methods: Logistic Regression, KNN, Naive Bayes, Classification Tree, Random Forest
###############################
n <- nrow(augmentedData)
index <- sample(1:n, n*0.8, replace=FALSE)
trainDat <- augmentedData[index, ]
testDat <- augmentedData[-index, ]

# Cross fold validation
nFolds <- 5
folds <- createFolds(trainDat, k = nFolds)


# Not this one
###############################
# KNN
###############################
set.seed(1)
train.error <- c()
test.error <- c()
for(k in 1:50) {
  train.error.temp <- c()
  test.error.temp <- c()
  # % fold cross validation
  for(i in 1:nFolds) {
    model.knn.train <- knn(train = trainDat[-folds[[i]], -19], test = trainDat[-folds[[i]], -19], cl=trainDat[-folds[[i]], 19], k=k)
    train.error.temp <- c(train.error.temp, mean(model.knn.train != trainDat[-folds[[i]], 19]))
    model.knn.test <- knn(train = trainDat[-folds[[i]], -19], test = trainDat[folds[[i]], -19], cl=trainDat[-folds[[i]], 19], k=k)
    test.error.temp <- c(test.error.temp, mean(model.knn.test != trainDat[-folds[[i]], 19]))
  }
  train.error = rbind(train.error,train.error.temp)
  test.error = rbind(test.error,test.error.temp)
}
# Plotting
par(mfrow = c(1,1))
plot(1:50, rowMeans(train.error), col='blue', type='b', ylim = c(0,0.4))
points(1:50, rowMeans(test.error), col='red', type='b')
# legend(1, 95, legend = c("Training MSE", "Testing MSE"), col = c("blue", "red"), title = "Legend")
best_k <- which.min(rowMeans(test.error))
best_k

# Test set
model.knn.pred = knn(train=testDat[, -19], test=testDat[, -19], cl=testDat[, 19], k=best_k)
# checking results
table(model.knn.pred, testDat[, 19])

model.knn.pred2 = knn(train=testDat[, -19], test=testDat[, -19], cl=testDat[, 19], k=1)
model.knn.pred3 = knn(train=testDat[, -19], test=testDat[, -19], cl=testDat[, 19], k=5)
model.knn.pred4 = knn(train=testDat[, -19], test=testDat[, -19], cl=testDat[, 19], k=10)

pred_knn <- prediction(as.numeric(model.knn.pred), testDat[, 19])
perf <- performance(pred_knn, "tpr", "fpr")

pred_knn2 <- prediction(as.numeric(model.knn.pred2), testDat[, 19])
perf2 <- performance(pred_knn2, "tpr", "fpr")

pred_knn3 <- prediction(as.numeric(model.knn.pred3), testDat[, 19])
perf3 <- performance(pred_knn3, "tpr", "fpr")

pred_knn4 <- prediction(as.numeric(model.knn.pred4), testDat[, 19])
perf4 <- performance(pred_knn4, "tpr", "fpr")

plot(perf, avg="threshold", colorize = TRUE)
plot(perf2, add = TRUE, avg="threshold", colorize = TRUE)
plot(perf3, add = TRUE, avg="threshold", colorize = TRUE)
plot(perf4, add = TRUE, avg="threshold", colorize = TRUE)


###############################
# Naive Bayes
###############################
train_control <- trainControl(method = "cv", number = 5)
model.nb <- train(trainDat[, -19], trainDat[, 19], 'nb', trControl = trainControl(method = 'cv', number = 5))

augmentedData <- cleanData
augmentedData <- augmentedData[!(augmentedData$id %in% c(53)), ]
augmentedData$age <- as.numeric(augmentedData$age)
augmentedData <- augmentedData[ , !(names(augmentedData) %in% c('id', 'result'))]
for(i in 1:10) {
  augmentedData[ ,i] <- as.factor(augmentedData[ ,i])
}
augmentedData$Class.ASD <- as.numeric(as.factor(augmentedData$Class.ASD)) - 1
#check
str(augmentedData)



newData <- cleanData[, c(12:16, 22)]
newData <- cbind(newData, pr.out$x[, 1:3])
#newData <- newData[!(newData$id %in% c(53)), ]

LR.model <- glm(Class.ASD ~., family = binomial("logit"), newData)
summary(LR.model)

table(augmentedData$age)

