# Linear discriminant analysis on stock market
library(ISLR)

setwd('/home/williamn/Repository/finance')

# Loading data
data(Default)
attach(Default)
names(Default)

head(Default)

# Estimating muk, pik, and variance

# Converting non-numeric features to numeric
levels(Default[, 2])[levels(Default[, 2]) == "No"] <- "0"
levels(Default[, 1])[levels(Default[, 1]) == "Yes"] <- "1"
Default[, 2] <- as.numeric(Default[, 2])

k <- 2
X <- Default[,2:4]
X <- data.matrix(X)
Y <- Default[,1]
Y <- data.matrix(Y)
mu <- data.frame()
pi <- c()
for(i in 1:k){
    tmp <- Default[Default$default == i,]
    mu <- rbind(mu, sapply(tmp[,2:dim(tmp)[2]], mean))
    pi <- rbind(pi, dim(tmp)[1]/dim(Default)[1])
}
sigma <- cov(X)

mu1 <- mu[1,]
mu2 <- mu[2,]
mu1 <- data.matrix(mu1)
mu2 <- data.matrix(mu2)
delta1 <- X%*%solve(sigma)%*%t(mu1) - as.numeric(0.5*mu1%*%solve(sigma)%*%t(mu1)) + log(pi[1])
delta2 <- X%*%solve(sigma)%*%t(mu2) - as.numeric(0.5*mu2%*%solve(sigma)%*%t(mu2)) + log(pi[2])
delta <- c()
for(j in 1:dim(Default)[1]){
    if(delta1[j] >= delta2[j]){
        delta <- rbind(delta, c(1))
    }else{
        delta <- rbind(delta, c(2))
    }
}


