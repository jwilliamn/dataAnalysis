corr<- function(directory, threshold = 0){
    setwd("/home/williamn/Repository/dataAnalysis/airPollution")
    listFiles<- list.files(directory, full.names = TRUE)
    source('complete.R')
    data<- complete("specdata", 1:332)
    corData<- data[data$nobs>threshold,]
    corf<- c()
    if(dim(corData)[1]>0){
        for(i in 1:dim(corData)[1]){
            tmp<- read.csv(listFiles[corData$id[i]])
            tmp_complete<- complete.cases(tmp)
            tmp_cor<- tmp[tmp_complete,]
            corf<- c(corf, cor(tmp_cor$sulfate, tmp_cor$nitrate))
        }
    }
    else {
        corf<- numeric()
    }
    corf
}