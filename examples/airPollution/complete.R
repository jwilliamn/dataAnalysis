complete<- function(directory, id=1:332){
    setwd("/home/williamn/Repository/dataAnalysis/airPollution")
    listFiles<- list.files(directory, full.names = TRUE)
    dat<- data.frame()
    for(i in id){
        tmp<- read.csv(listFiles[i])
        good_tmp<- complete.cases(tmp)
        complete<- tmp[good_tmp,]
        nobs<- dim(complete)
        dat<- rbind(dat, c(id=i, nobs=nobs[1]))
    }
    colnames(dat)<- c("id", "nobs")
    dat
}