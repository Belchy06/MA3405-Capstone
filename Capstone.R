library(ggplot2)
library(ggfortify)
library(dplyr)
library(class)
library(caret)
library(e1071)

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

# Age has a maximum value of 383. This is definitely an outlier, should be removed
augmentedData <- augmentedData[!(augmentedData$id %in% c(53)), ]
augmentedData <- augmentedData[ , !(names(augmentedData) %in% c('id', 'result'))]



checked_data <- augmentedData %>%
  rowwise() %>%
  mutate(question_sum = sum(c(A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score))) 
mismatched_data <- checked_data[checked_data$result != checked_data$question_sum, ]


###############################
# PCA
###############################
pca.dat <- augmentedData[ , !(names(augmentedData) %in% c('Class.ASD'))]
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
biplot(pr.out, scale = 0)
autoplot(pr.out, colour = 'gender', shape = FALSE, label.size = 3)

# variance explained by each PC
pr.var <- pr.out$sdev^2
# proportion of variance explained
pve <- pr.var / sum(pr.var)
# plot PVE
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")


###############################
# Possible methods: Logistic Regression, KNN, Naive Bayes, 
###############################
n <- nrow(augmentedData)
index <- sample(1:n, n*0.8, replace=FALSE)
trainDat <- augmentedData[index, ]
testDat <- augmentedData[-index, ]

# Cross fold validation
nFolds <- 5
folds <- createFolds(trainDat, k = nFolds)



###############################
# KNN
###############################
set.seed(2)
train.error <- c()
test.error <- c()
for(k in 1:30) {
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
plot(1:30, rowMeans(train.error), col='blue', type='b', ylim = c(0,0.4))
points(1:30, rowMeans(test.error), col='red', type='b')
legend(1, 95, legend = c("Training MSE", "Testing MSE"), col = c("blue", "red"), title = "Legend")

# Test set
model.knn.pred = knn(train=testDat[, -19], test=testDat[, -19], cl=testDat[, 19], k=11)
# checking results
table(model.knn.pred, testDat[, 19])
accuracy.k11 = mean(model.knn.pred == testDat[, 19])
error.rate.k11 = mean(model.knn.pred != testDat[, 19])
accuracy.k11
error.rate.k11

###############################
# Naive Bayes
###############################
train_control <- trainControl(method = "cv", number = 5)
model.nb <- train(trainDat[, -19], trainDat[, 19], 'nb', trControl = trainControl(method = 'cv', number = 5))

