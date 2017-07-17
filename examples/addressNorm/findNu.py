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
import string

from difflib import SequenceMatcher

import sys


# Function definitions
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Main function ####
if __name__ == '__main__':
    """ .........
    To run the app, execute the following in terminal:

    [terminal_prompt]$ python findNu.py path/to/file.csv

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
    findData = pd.read_csv(argLook) #, encoding='cp1250')


    
    # A little bit of exploration of Data
    #print(data)
    #print(len(data["DIRECCION"].unique()), type(data))
    #print(data.columns)
    #print(data.index)
    #print(data.head(10))
    print("Original shape: ", data.shape)
    #print(data["DIRECCION"].describe())

    #newData = data.iloc[:,[0,1,2,3,4,5,6,7,51,52,53,54,55,56,57,58,59,60,61]].copy()
    newData = data[["TIPO_DOCUMENTO","NUMERO_DOCUMENTO","APELLIDO_PATERNO","APELLIDO_MATERNO",
                        "NOMBRES","FECHA_NACIMIENTO","SEXO","DIRECCION","TIPO_NUCLEO _URBANO",
                        "NUCLEO_URBANO","TIPO_VIA","NOMBRE_VIA","NUMERO_PUERTA","BLOCK","PISO",
                        "INTERIOR","MANZANA","LOTE","KILOMETRO"]].copy()


    # Search of District based on "Nucleo Urbano"
    newData["DISTRICT_"] = pd.Series("", index = data.index)
    newData["MATCH_"] = pd.Series("None", index = newData.index)

    # To upper
    for d in range(0, findData.shape[0]):
        findData.loc[d, "NOMBRE N.U"] = findData["NOMBRE N.U"][d].upper()
        findData.loc[d, "NOMBRE N.U"] = findData["NOMBRE N.U"][d].strip()

    
    # Perfect match
    newData.loc[newData["NUCLEO_URBANO"].isin(findData["NOMBRE N.U"]), "DISTRICT_"] = argVal
    newData.loc[newData["DISTRICT_"] == argVal, "MATCH_"] = "Perfect_match"

    # Filling missing target column
    newData["NUCLEO_URBANO"] = newData["NUCLEO_URBANO"].fillna("MissingNu")
    
    # Clean up punctuation
    """
    for p in range(0, newData.shape[0]):
        if newData["DISTRICT_"][p] == "":
            # Evaluation (It could be ommited depending on performance)
            translator = newData["NUCLEO_URBANO"][p].maketrans('','', string.punctuation)
            newData.loc[p, "NUCLEO_URBANO"] = newData["NUCLEO_URBANO"][p].translate(translator)
    """

    # Included and partially matched
    for i in range(0, newData.shape[0]):
        print("______Nu Urb: ", i)
        for j in range(0, findData.shape[0]):
            if (((newData["NUCLEO_URBANO"][i] in findData["NOMBRE N.U"][j]) or
             (findData["NOMBRE N.U"][j] in newData["NUCLEO_URBANO"][i])) and 
            newData["DISTRICT_"][i] == ""):
                newData.loc[i,"DISTRICT_"] = argVal
                newData.loc[i,"MATCH_"] = "Included"
            if (newData["DISTRICT_"][i] == "" and 
                similarity(newData["NUCLEO_URBANO"][i], findData["NOMBRE N.U"][j]) > 0.85):
                newData.loc[i,"DISTRICT_"] = argVal
                newData.loc[i,"MATCH_"] = "Partial_match"
            
        

    # Write output to file
    newData.to_csv("dataVentFound.csv", index=False)

    