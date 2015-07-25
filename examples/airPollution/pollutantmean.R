pollutantmean<- function(directory, pollutant, id=1:332){
    setwd("/home/williamn/Repository/dataAnalysis/airPollution")
    listFiles<- list.files(directory, full.names = TRUE)
    dat<- data.frame()
    for(i in id){
        dat<- rbind(dat, read.csv(listFiles[i]))
    }
    #tmp<- lapply(listFiles, read.csv)
    #output<- do.call(rbind. tmp)
    
    # Subsetting
    #median(dat$pollutant, na.rm = TRUE)
    mean(dat[,pollutant], na.rm = TRUE)
}