library(ggplot2)
library(ggfortify)


importedData <- read.csv(file = "csv_result-Autism-Adult-Data.csv")
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
augmentedData <- augmentedData[ , !(names(augmentedData) %in% c('id'))]


# PCA
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
pve
# plot PVE
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")

################################
# Heirachal Clustering to see if clusters exist in the data
################################
hc.complete <- hclust(dist(pca.dat), method = "complete")
hc.average <- hclust(dist(pca.dat), method = "average")
hc.single <- hclust(dist(pca.dat), method = "single")
par(mfrow = c(1,3))
plot(hc.complete, main="Complete Linkage", xlab="", sub="", cex=.9)
plot(hc.average, main="Average Linkage", xlab="", sub="", cex=.9)
plot(hc.single, main="Single Linkage", xlab="", sub="", cex=.9)
