# Problem set Nº 3 from the Data Analysis with R

setwd('/home/williamn/Repository/dataAnalysis/udacity/data\ analysis\ R')

# Loading necessary packages
library(ggplot2)
library(ggthemes)
data(diamonds)
dim(diamonds)
str(diamonds)

# Histogram of the price of all the diamonds
qplot(x = price, data = diamonds, 
      color = I('black'), fill = I('#099DD9')) + 
    theme_economist() +
    scale_colour_economist() +
    ggtitle("Diamonds' price")

# How many dimonds cost less than ?
d500<- sum(diamonds$price < 500); d250<- sum(diamonds$price < 250);
d15000<- sum(diamonds$price >= 15000)

# Limiting the x-axis, altering the bin width, and setting different breaks
# on the x-axis
ggplot(aes(x = price), data = diamonds) + 
    geom_histogram(binwidth = 100, color = I('black'), fill = I('#099DD9')) + 
    scale_x_continuous(limits = c(0, 10000), breaks = seq(0, 10000, 500)) + 
    theme_economist() + 
    scale_colour_economist() + 
    ggtitle("Diamonds' price")

# Price by cut histograms with free "y" scale (not fixed)
ggplot(aes(x = price), data = diamonds) + 
    geom_histogram(binwidth = 100, color = I('black'), fill = I('#099DD9')) + 
    scale_x_continuous(limits = c(0, 10000), breaks = seq(0, 10000, 1000)) + 
    facet_wrap(~cut, scales = "free_y")
ggsave("diamonds.png")

# Which cut has the highest priced diamond?
by(diamonds$price, diamonds$cut, max)

# Adding different effects for a better visualization
ggplot(aes(x = price), data = diamonds) + 
    geom_histogram(binwidth = 100, color = I('black'), fill = I('#099DD9'), 
                   aes(y = ..density..)) + 
    geom_density() + 
    scale_x_continuous(limits = c(0, 10000), breaks = seq(0, 10000, 1000)) + 
    facet_wrap(~cut, scales = "free_y")
