__author__ = 'williamn'
# Introduction to computational thinking and data science - 6.00.2x

import pylab
import string
import numpy as np

# example
# print 'I like 6.00.2x!'
# principal = 10000 #initial investment
# interestRate = 0.05
# years = 20
# values = []
# for i in range(years + 1):
#     values.append(principal)
#     principal += principal*interestRate
# pylab.plot(range(years+1), values)
# pylab.title('5% Growth, Compounded Annually')
# pylab.xlabel('Years of Compounding')
# pylab.ylabel('Value of Principal ($)')
# pylab.show()

# L1 Problem 3
path_to_file='/home/williamn/Repository/dataScience/python/julyTemps.txt'
highTemp=[]
lowTemp=[]

def loadData():
    inFile=open(path_to_file)
    for line in inFile:
        fields=line.split()
        if len(fields) != 3 or 'Boston' == fields[0] or 'Day' == fields[0]:
            continue
        else:
            highTemp.append(int(fields[1]))
            lowTemp.append(int(fields[2]))
    print highTemp
    return (lowTemp, highTemp)

(low, high)=loadData()

def producePlot(lowTemps, highTemps):
    # diffTemps=[]
    # for i in range(len(lowTemp)):
    #     diffTemps.append(highTemp[i]-lowTemp[i])
    # Code using Numpy
    diffTemps=list(np.array(highTemps)-np.array(lowTemps))
    pylab.plot(range(1,32), diffTemps)
    pylab.title('Day by Day Ranges in Temperature in Boston in July 2012')
    pylab.xlabel('Days')
    pylab.ylabel('Temperature Ranges')
    pylab.show()

producePlot(low, high)