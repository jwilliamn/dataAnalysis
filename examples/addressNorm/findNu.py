#!/usr/bin/env python
# coding: utf-8

"""
    findNu
    ============================

    Find "Nucleo Urbano"
        Find the District to which the NU belongs to 

    _copyright_ = 'Copyright (c) 2017 J.W.'
"""

import numpy as np
import pandas as jhon
import pandas as pd

import sys


# Function definitions
def searchAdd(addrId, address):
    posId = -1
    for x in addrId:
        if address.find(x) != -1:
            posId = address.find(x)
            print(x, " found! ", posId)
            break
    return posId


# Main function ####
if __name__ == '__main__':
    """ .........
    To run the app, execute the following in terminal:

    [terminal_prompt]$ python addrNorm.py path/to/file.csv

    Currently the app supports files in the following formats: 
        .csv
    """
    RED='\033[0;31m'
    OKBLUE = '\033[5;94m'
    OKGREEN = '\033[1;92m'
    #OKGREEN = '\x1b[1;92m'
    NC='\033[0m' # No Color

    print(OKGREEN + "Init of Search!" + NC)

    # Get data and params from terminal
    arg = sys.argv[1]
    argLook = sys.argv[2]
    argVal = sys.argv[3]
    
    # Read data
    data = pd.read_csv(arg)
    dataFind = pd.read_csv(argLook) #, encoding='cp1250')


    
    # A little bit of exploration of Data
    #print(data)
    #print(len(data["DIRECCION"].unique()), type(data))
    #print(data.columns)
    #print(data.index)
    #print(data.head(10))
    print("Original shape: ", data.shape)
    #print(data["DIRECCION"].describe())

    newData = data.iloc[:,[0,1,2,3,4,5,6,7,52,53,54,55,56,57,58,59,60,61]].copy()


    # Search of District based on "Nucleo Urbano"
    newData["DISTRICT_"] = pd.Series("", index = data.index)

    # uhh
    for d in range(0, dataFind.shape[0]):
        dataFind.loc[d, "NOMBRE N.U"] = dataFind["NOMBRE N.U"][d].upper()

    print(dataFind.head(10))

    newData["DISTRICT_"].loc[newData["NUCLEO_URBANO"].isin(dataFind["NOMBRE N.U"])] = argVal
    #for i in range(0, newData.shape[0]):
        #if newData["NUCLEO_URBANO"][i].isin(dataFind["NOMBRE N.U"]):
        #    newData.loc[i,"DISTRICT_"] = argVal
        #newData.loc[i,"DISTRICT_"].loc[newData["NUCLEO_URBANO"].isin(dataFind["NOMBRE N.U"])] = argVal
        #newData["DISTRICT_"].loc[newData["NUCLEO_URBANO"].isin(dataFind["NOMBRE N.U"].upper())] = argVal


    # Write output to file
    newData.to_csv("dataVentFound.csv", index=False)

    