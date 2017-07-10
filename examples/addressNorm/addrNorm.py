#!/usr/bin/env python
# coding: utf-8

"""
    addrNorm
    ============================

    Address normalizations script
        Addresses of Peruvian households "sometimes" are different on different Goverment Data bases

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

def searchDic(dictionary):
    tipLabel = "NA"
    posValue = -1
    for label in dictionary:
        if dictionary[label] != -1:
            tipLabel = label
            posValue = dictionary[label]
            break

    #print("Label  %s \tPosi %d" % (tipLabel, posValue))
    return tipLabel, posValue

def searchNum(weirdAddr):
    num = [s for s in weirdAddr.split() if s.isdigit()]
    numpos = searchAdd(num, weirdAddr)
    if len(num) > 0:
        return numpos, num[0], len(num[0])
    else:
        return numpos, None, None

def cleanPref(pattern, dirtyAddr):
    cleanAddr = dirtyAddr
    for p in range(0, len(pattern)):
        if dirtyAddr.find(pattern[p]) != -1:
            cleanAddr = dirtyAddr.replace(pattern[p], "")
            #print("clean: ", cleanAddr)
            break

    return cleanAddr


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

    print(OKGREEN + "Init of Magic!" + NC)

    # Get data and params from terminal
    arg = sys.argv[1]
    addrCol = sys.argv[2]
    fileOut = sys.argv[3]
    
    # Read data
    data = pd.read_csv(arg)

    
    # A little bit of exploration of Data
    #print(data)
    #print(len(data["DIRECCION"].unique()), type(data))
    #print(data.columns)
    #print(data.index)
    #print(data.head(10))
    print("Original shape: ", data.shape)
    #print(data["DIRECCION"].describe())


    # Creation of new variables    
    data["TIPO_NUCLEO _URBANO"] = pd.Series("", index = data.index)
    data["NUCLEO_URBANO"] = pd.Series("", index = data.index)
    data["TIPO_VIA"] = pd.Series("", index = data.index)
    data["NOMBRE_VIA"] = pd.Series("", index = data.index)
    data["NUMERO_PUERTA"] = pd.Series("", index = data.index)
    data["BLOCK"] = pd.Series("", index = data.index)
    data["PISO"] = pd.Series("", index = data.index)
    data["INTERIOR"] = pd.Series("", index = data.index)
    data["MANZANA"] = pd.Series("", index = data.index)
    data["LOTE"] = pd.Series("", index = data.index)
    data["KILOMETRO"] = pd.Series("", index = data.index)


    # Analysis of Addresses ####
    #data["DIRECCION"] = data["DIRECCION"].fillna("Missing Addr")
    #addr = data["DIRECCION"]
    data[addrCol] = data[addrCol].fillna("Missing Addr")
    addr = data[addrCol]

    # Address schemes ####
    # Nucleo Urbano
    #city = ["CIUDAD"]
    upis = ["UPIS"]
    asen = ["AAHH", "AA HH", "A.A.H.H.", "ASENT H", "AH", "ASENTAMIENTO HUMANO", 
            "AA  HH", "ASENT. H.", "ASENT.H.", "ASENT.HUMANO", 
            "ASENT. HUMANO", "AH.", "AA.HH","A.H", "A.H.", "ASEN H", "AA    HH", "AA   HH",
            "AA JJ", "AA ", "aa hh", "aahh", "ah", "AJ ", "ASEN HUMAN", "ASENTA ", "ASENT ",
            "ASENTE H", "ASENTH", "ASET H", "APV ", "ASNT H", "ASEN HUMA", "ASEN T H", 
            "ASN ", "ASENTE H", "a h ", "A  H", "A H"]
    pueJ = ["PPJJ", "PJ.", "PJ", "PP JJ", "PUEBLO JOVEN", "P.JOVEN", "P. JOVEN", " P J "]
    urba = ["URBANIZACION", "URBANIZ", "URB.", "URBAN", "URB", "UR."] # ENACE
    pueb = ["PUEBLO"]
    case = ["CASERIO"]
    anex = ["ANEXO"]
    coop = ["COOPERATIVA AGRARIA"]
    camp = ["CAMPAMENTO MINERO"]
    conj = ["CONJUNTO HABITACIONAL", "CONJ.HAB.", "CONJ HAB", "C HAB "]
    asoc = ["ASOCIACION", "ASOCIACIÓN", "ASOC"]
    cooV = ["COOPERATIVA DE VIVIENDA"]
    barr = ["BARRIO"]
    ccpp = ["CENTRO POBLADO", "CC.PP"]


    # TIPO DE VIA 
    call = ["CALLE", "CLL.", "CLL", "CAL ", " CA "] 
    aven = ["AV.", "AVENIDA", "AVE ", "AV "]
    jron = ["JR.", "JR", "JIRON"]
    carr = ["CARRETERA"]
    psje = ["PJE", "PSJE.", "PSJE", "PASAJE"]

    # NOMBRE DE VIA  
    # Número de puerta
    num = ["NUMERO", "N°", "NRO", " N ", "SN "]

    # BLOCK 
    blck = ["BLOCK", "BLOQ ", "BLOK", " BL ", "BQ "] 

    # PISO  
    piso = ["PISO"]

    # INTERIOR   
    inte = ["INTERIOR", "INT.", "INTE", "INT", "DPT.", "DPTO.", "DEPA", "DPT", "DPTO"]

    # MANZANA 
    mzna = ["MZA", "MZ.", "MZ", "MAZ", "MANZANA", "MANZ"]

    # LOTE 
    lote = ["LT", "LT.", "LOTE", "LTE", "LOT"]

    # KILOMETRO

    
    # Pos of address identifiers ####
    for j in range(0, len(addr)):

        indAddr = addr[j]
        print(OKBLUE + "Address: " + NC, j, indAddr)

        upispos = searchAdd(upis, indAddr)
        asenpos = searchAdd(asen, indAddr)
        pueJpos = searchAdd(pueJ, indAddr)
        urbapos = searchAdd(urba, indAddr)
        puebpos = searchAdd(pueb, indAddr)
        casepos = searchAdd(case, indAddr)
        anexpos = searchAdd(anex, indAddr)
        cooppos = searchAdd(coop, indAddr)
        camppos = searchAdd(camp, indAddr)
        conjpos = searchAdd(conj, indAddr)
        asocpos = searchAdd(asoc, indAddr)
        cooVpos = searchAdd(cooV, indAddr)
        barrpos = searchAdd(barr, indAddr)
        ccpppos = searchAdd(ccpp, indAddr)


        callpos = searchAdd(call, indAddr)
        avenpos = searchAdd(aven, indAddr)
        jronpos = searchAdd(jron, indAddr)
        carrpos = searchAdd(carr, indAddr)
        psjepos = searchAdd(psje, indAddr)

        numpos = searchAdd(num, indAddr)

        blckpos = searchAdd(blck, indAddr)

        pisopos = searchAdd(piso, indAddr)

        intepos = searchAdd(inte, indAddr)

        mznapos = searchAdd(mzna, indAddr)

        lotepos = searchAdd(lote, indAddr)


        # Dictionary of Identifiers
        nuclUrb = {"UPIS":upispos, "ASENTAMIENTO HUMANO":asenpos, "PUEBLO JOVEN":pueJpos, 
        "URBANIZACION":urbapos, "PUEBLO":puebpos, "CASERIO":casepos, "ANEXO":anexpos, 
        "COOPERATIVA AGRARIA":cooppos, "CAMPAMENTO MINERO":camppos, "CONJUNTO HABITACIONAL":conjpos, 
        "ASOCIACION DE VIVIENDA":asocpos, "COOPERATIVA DE VIVIENDA":cooVpos, "BARRIO":barrpos, 
        "CENTRO POBLADO":ccpppos}

        tipoVia = {"CALLE":callpos, "AVENIDA":avenpos, "JIRON":jronpos, "CARRETERA":carrpos, 
        "PASAJE":psjepos}

        numPuer = {"NUMERO":numpos}

        blockDi = {"BLOCK":blckpos}

        pisoDic = {"PISO":pisopos}

        interDi = {"INTERIOR":intepos}

        manzDic = {"MANZANA":mznapos}

        loteDic = {"LOTE":lotepos}

        tipoNuc, posNucl = searchDic(nuclUrb)
        tipoVia, posVia = searchDic(tipoVia)
        tipoPue, posPuer = searchDic(numPuer)
        tipoBlo, posBlock = searchDic(blockDi)
        tipoPis, posPiso = searchDic(pisoDic)
        tipoInt, posInter = searchDic(interDi)
        tipoMan, posManz = searchDic(manzDic)
        tipoLot, posLote = searchDic(loteDic)

        # Whole structure
        addrStructure = {tipoNuc:posNucl, tipoVia:posVia, tipoPue:posPuer, tipoBlo:posBlock, 
        tipoPis:posPiso, tipoInt:posInter, tipoMan:posManz, tipoLot:posLote}

        addrStructure = {k: v for k, v in addrStructure.items() if k != "NA"}
        print(addrStructure)

        keyOrd = sorted(addrStructure, key=addrStructure.get)

        # Split address according to structure
        listNuc = {"UPIS", "ASENTAMIENTO HUMANO", "PUEBLO JOVEN", "URBANIZACION", "PUEBLO", "CASERIO",
        "ANEXO", "COOPERATIVA AGRARIA", "CAMPAMENTO MINERO", "CONJUNTO HABITACIONAL", 
        "ASOCIACION DE VIVIENDA", "COOPERATIVA DE VIVIENDA", "BARRIO", "CENTRO POBLADO"}
        listVia = {"CALLE", "AVENIDA", "JIRON", "CARRETERA", "PASAJE"}

        varTnu = ""
        varNuc = ""
        varTvi = ""
        varVia = ""
        varNum = ""
        varBlo = ""
        varPis = ""
        varInt = ""
        varMan = ""
        varLot = ""
        varKmt = ""

        for i, val in enumerate(addrStructure):
            if keyOrd[i] in listNuc:
                varTnu = keyOrd[i]
                if i == (len(addrStructure) -1):
                    varNuc = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varNuc = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varNuc)

            elif keyOrd[i] in listVia and i < len(addrStructure):
                varTvi = keyOrd[i]
                if i == (len(addrStructure) -1):
                    varVia = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varVia = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varVia)

            elif keyOrd[i] in {"NUMERO"} and i < len(addrStructure):
                if i == 0:
                    varVia = indAddr[0:addrStructure[keyOrd[i]]]

                if i == (len(addrStructure) -1):
                    varNum = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varNum = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varNum)
            elif keyOrd[i] in {"BLOCK"} and i < len(addrStructure):
                if i == (len(addrStructure) -1):
                    varBlo = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varBlo = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varBlo)
            elif keyOrd[i] in {"PISO"} and i < len(addrStructure):
                if i == (len(addrStructure) -1):
                    varPis = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varPis = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varPis)
            elif keyOrd[i] in {"INTERIOR"} and i < len(addrStructure):
                if i == 0:
                    varVia = indAddr[0:addrStructure[keyOrd[i]]]

                if i == (len(addrStructure) -1):
                    varInt = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varInt = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varInt)
            elif keyOrd[i] in {"MANZANA"} and i < len(addrStructure):
                if i == 0:
                    varNuc = indAddr[0:addrStructure[keyOrd[i]]]

                if i == (len(addrStructure) -1):
                    varMan = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varMan = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]
                print(varMan)
            elif keyOrd[i] in {"LOTE"} and i < len(addrStructure):
                if i == (len(addrStructure) -1):
                    varLot = indAddr[addrStructure[keyOrd[i]]:]
                else:
                    varLot = indAddr[addrStructure[keyOrd[i]]:addrStructure[keyOrd[i+1]]]

                # Check Lote
                if varLot != "":
                    posnumV_, singleNum_, lenNum_ = searchNum(varLot)
                    if posnumV_ != -1 and len(varLot[(posnumV_ + lenNum_):]) > 0:
                        if varNuc == "":
                            varNuc = varLot[(posnumV_ + lenNum_):]
                        else:
                            varNuc = varNuc + " " + varLot[(posnumV_ + lenNum_):]
                        varLot = varLot[0:(posnumV_ + lenNum_)]                        

                print(varLot)

            else:
                continue

        if not addrStructure:
            print("Abandonned address: ", indAddr)
            posnumV, singleNum, lenNum = searchNum(indAddr)
            if posnumV == -1 and not singleNum :
                varTnu = "BE CAREFUL"
                varNuc = indAddr
            else:
                varVia = indAddr[0:posnumV]
                varNum = singleNum
                varNuc = indAddr[(posnumV + lenNum):]



        data.loc[j,"TIPO_NUCLEO _URBANO"] = varTnu
        data.loc[j,"NUCLEO_URBANO"] = varNuc
        data.loc[j,"TIPO_VIA"] = varTvi
        data.loc[j,"NOMBRE_VIA"] = varVia
        data.loc[j,"NUMERO_PUERTA"] = varNum
        data.loc[j,"BLOCK"] = varBlo
        data.loc[j,"PISO"] = varPis
        data.loc[j,"INTERIOR"] = varInt
        data.loc[j,"MANZANA"] = varMan
        data.loc[j,"LOTE"] = varLot
        data.loc[j,"KILOMETRO"] = varKmt

    
        # Get rid of non normalized address prefixes
        nucPref = (upis + asen + pueJ + urba + pueb + case + anex + coop + camp + conj + 
            asoc + cooV + barr + ccpp)

        viaPref = call + aven + jron + carr + psje
       
        data.loc[j,"NUCLEO_URBANO"] = cleanPref(nucPref, varNuc)
        data.loc[j,"NOMBRE_VIA"] = cleanPref(viaPref, varVia)
        data.loc[j,"NUMERO_PUERTA"] = cleanPref(num, varNum)
        data.loc[j,"BLOCK"] = cleanPref(blck, varBlo)
        data.loc[j,"PISO"] = cleanPref(piso, varPis)
        data.loc[j,"INTERIOR"] = cleanPref(inte, varInt)
        data.loc[j,"MANZANA"] = cleanPref(mzna, varMan)
        data.loc[j,"LOTE"] = cleanPref(lote, varLot)


    # Write output to file
    #data.to_csv("dataSep.csv", index=False)
    data.to_csv(fileOut, index=False)
    
    """
    print("Nucleo urbano found: ", upispos, asenpos, pueJpos, urbapos, puebpos, casepos, anexpos, 
        cooppos, camppos, conjpos, asocpos, cooVpos, barrpos, ccpppos)
    print("Tipo de via found: ", callpos, avenpos, jronpos, carrpos, psjepos)
    print("Número found: ", numpos)
    print("Block found: ", blckpos)
    print("Piso found: ", pisopos)
    print("Interior found: ", intepos)
    print("Manzana found: ", mznapos)
    print("Lote found: ", lotepos)
    """