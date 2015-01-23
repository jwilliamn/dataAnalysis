# Working with files

setwd("/home/williamn/Repository/dataAnalysis/dietAnalysis")

list.files("diet_data")
andy<- read.csv('diet_data/Andy.csv')
head(andy)
length(andy$Day)
dim(andy)
str(andy)
summary(andy)
names(andy)

# subseting
andy[1, "Weight"]
andy[which(andy$Day==30), "Weight"]
andy[which(andy[,"Day"]==30), "Weight"]

# Working with files
files_full<- list.files("diet_data", full.names = TRUE)
files_full
head(read.csv(files_full[3]))

# Binding files
andy_david<- rbind(andy, read.csv(files_full[2]))
head(andy_david)
tail(andy_david)
dim(andy_david)
day_25<- andy_david[which(andy_david$Day ==25),]
day_25

dat<- data.frame()
for(i in 1:5){
    dat<- rbind(dat,read.csv(files_full[i]))
}
median(dat$Weight, na.rm = TRUE)
dat_30<- dat[dat$Day==30,]
dat_30
median(dat_30$Weight)

# Functions
weightMedian<- function(directory, day){
    files_list<- list.files(directory, full.names = TRUE)  # Creates a list of files
    dat<- data.frame()  # Creates an empty frame
    for(i in 1:5){
        dat<- rbind(dat, read.csv(files_list[i]))
    }
    dat_subset<- dat[dat$Day==day,]
    median(dat_subset$Weight, na.rm = TRUE)
}