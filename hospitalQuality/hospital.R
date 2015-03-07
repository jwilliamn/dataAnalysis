# Programming assignment 3 - R Programming

setwd("/home/williamn/Repository/dataAnalysis/hospitalQuality")
outcome<- read.csv("outcome-of-care-measures.csv", colClasses = "character")

head(outcome)
library(ggplot2)

# 30 day death rates from heart attack histogram
outcome[, 11]<- as.numeric(outcome[, 11])
hist(outcome[, 11])

# Using qplot
qplot(x = outcome[, 11], data = outcome)

best<- function(state, outcome){
    data<- read.csv("outcome-of-care-measures.csv", colClasses = "character")
    state<- data[data$State == state,]
    ha<- "heart attack"
    hf<- "heart failure"
    pn<- "pneumonia"
    if(length(state$State) > 0){
        if(outcome != ha || outcome != hf || outcome != pn){
            stop("invalid outcome")
        }
        else{ stop("valid outcome")}
    }
    else{
        stop("invalid state")
    }
}
