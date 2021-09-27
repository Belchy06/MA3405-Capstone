##########################################
# Principal Component Analysis
##########################################

data <- USArrests
states <- rownames(data)
states
names(data)
# columnwise mean of data
apply(data, 2, mean)

# columnwise variance of data
apply(data, 2, var)

# PCA
pr.out <- prcomp(data, scale = TRUE)
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

# variance explained by each PC
pr.var <- pr.out$sdev^2
# proportion of variance explained
pve <- pr.var / sum(pr.var)
pve
# plot PVE
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0,1), type = "b")


##########################################
# Matrix Completion
##########################################
X <- data.matrix(scale(data))
pcob <- prcomp(X)
summary(pcob)
# SVD
# The matrix v is equivalent to the loading matrix from the principal components (flipped in this case)
sX <- svd(X)
names(sX)
round(sX$v, 3)

# The matrix u is equivalent to the matrix of standardized scores and the standard deviations are in the vector d
t(sX$d * t(sX$u))

# Randomly omit 20 entries from the 50 x 2 data matrix
nomit <- 20
set.seed(15)
ina <- sample(seq(50), nomit)
inb <- sample(1:4, nomit, replace = TRUE)
Xna <- X
index.na <- cbind(ina, inb)
Xna[index.na] <- NA
Xna

# Algorithm 12.1
fit.svd <- function(X, M = 1) {
  svdob <- svd(X)
  with(svdob,
       u[, 1:M, drop = FALSE] %*%
         (d[1:M] * t(v[, 1:M, drop = FALSE]))
       )
}

Xhat <- Xna
xbar <- colMeans(Xna, na.rm = TRUE)
Xhat[index.na] <- xbar[inb]


# Measure the progress of iterations
thresh <- 1e-7
rel_err <- 1
iter <- 0
ismiss <- is.na(Xna)
mssold <- mean((scale(Xna, xbar, FALSE)[!ismiss])^2)
mss0 <- mean(Xna[!ismiss]^2)

# Use Xapp to updates the estimates for elements in Xhat that are missing in Xna. Finally, compute the relative error
while(rel_err > thresh) {
  iter <- iter + 1
  # Step 2(a)
  Xapp <- fit.svd (Xhat, M = 1)
  # Step 2(b)
  Xhat[ismiss] <- Xapp[ismiss]
  # Step 2(c)
  mss <- mean(((Xna - Xapp)[!ismiss])^2)
  rel_err <- (mssold - mss) / mss0
  mssold <- mss
  cat("Iter :", iter, "MSS :", mss, "Rel.Err:", rel_err, "\n")
}

# Here, we implements Algorithm 12.1 for imputing missing values using PCA. Someone looking to do apply matrix completion to their own data should use the softImpute package on CRAN