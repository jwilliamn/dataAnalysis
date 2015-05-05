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
ggplot(aes(x = price, fill = cut), data = diamonds) + 
    geom_histogram(binwidth = 200, color = I('black')) + 
    scale_x_continuous(limits = c(0, 10000), breaks = seq(0, 10000, 1000)) + 
    facet_wrap(~cut, scales = "free_y")

# Price per carat by cut
ggplot(aes(x = price/carat), data = diamonds) + 
    geom_histogram(binwidth = 0.05, color = I('black'), fill = I('#099DD9')) + 
    scale_x_log10() + 
    facet_wrap(~cut, scales = "free_y")

# Price of diamonds using box plots
# Price by cut
ggplot(data = diamonds, aes(x = cut, y = price, fill = cut)) + 
    geom_boxplot() + 
    coord_cartesian(ylim = c(0, 8000))
by(diamonds$price, diamonds$cut, summary)

# Price by clarity
ggplot(data = diamonds, aes(x = clarity, y = price, fill = clarity)) + 
    geom_boxplot() + 
    coord_cartesian(ylim = c(0, 8000))
by(diamonds$price, diamonds$clarity, summary)

# Price by color
ggplot(data = diamonds, aes(x = color, y = price, fill = color)) + 
    geom_boxplot() + 
    coord_cartesian(ylim = c(0, 8000))
by(diamonds$price, diamonds$color, summary)

# Calc. IQR
IQR(subset(diamonds, color == "D")$price)
IQR(subset(diamonds, color == "J")$price)

# Price per carat box plots by color
ggplot(data = diamonds, aes(x = color, y = price/carat, fill = color)) + 
    geom_boxplot() + 
    coord_cartesian(ylim = c(0, 8000))
by(diamonds$price/diamonds$carat, diamonds$color, summary)

# Ralationship among price, carat and color of diamonds
ggplot(data=diamonds, aes(x=carat, y=price, col=color)) + 
    theme_bw() + 
    geom_point(size = 2, alpha = 1) +
    scale_colour_brewer(palette="Accent") +
    ggtitle("The relationship among 'price', 'carat' and 'color' of diamonds")

# Carat frequency polygon

