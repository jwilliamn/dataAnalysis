# Linear discriminant analysis - Default on credit card payments
library(ISLR)

# Loading data
data(Default)
attach(Default)
names(Default)

Data <- Default
rm(Default)

# Data pre-processing
## Converting non-numeric features to numeric
isf <- sapply(Data, is.factor)
for(c in 1:dim(Data)[2]){
    if(isf[c] == T){
        Data[,c] <- as.numeric(Data[,c])
    }
}

## Identifying the predictors X, the response Y & size of the target data
### number of the column response Y
cy <- 1

X <- data.matrix(Data[,2:3])
Y <- data.matrix(Data[,cy])
m <- dim(X)[1]
n <- dim(X)[2]

# Estimating muk, pik and the covariance
k <- 2
mu <- c()
sigma <- c(0)
pi <- c()
for(i in 1:k){
    subData <- Data[Data[,cy] == i,]
    Xi <- data.matrix(subData[,2:3])
    mu <- rbind(mu, sapply(subData[,2:3], mean))
    mui <- matrix(rep(mu[i,], each=dim(Xi)[1]), nrow = dim(Xi)[1])
    sigma <- sigma + t(Xi - mui)%*%(Xi - mui)/(m-k)
    pi <- rbind(pi, dim(Xi)[1]/m)
}

# Computing the discriminant function delta^
delta <- c()
ld <- c()
for(d in 1:k){
    mui <- matrix(mu[d,], nrow = 1)
    di <- X%*%solve(sigma)%*%t(mui) - as.numeric(0.5*mui%*%solve(sigma)%*%t(mui)) + log(pi[d])
    ld <- cbind(ld, solve(sigma)%*%t(mui))
    delta <- cbind(delta, di)
}

# Classifying the observations
lda.fit <- c()
for(i in 1:m){
    if(delta[i,1] >= delta[i,2]){
        lda.fit <- rbind(lda.fit, c(1))
    }else{
        lda.fit <- rbind(lda.fit, c(2))
    }
}

table(lda.fit)

# Computing the coefficients of linear discriminants (decision boundary between classes)

solve(sigma)%*%t(matrix(mu[2,], nrow = 1) - matrix(mu[1,], nrow = 1))



