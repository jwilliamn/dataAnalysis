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

import sys


# Function definitions
def searchAdd(addrId, address):
    #print("Address: ", address, type(address))

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

    print("Label  %s \tPosi %d" % (tipLabel, posValue))
    return tipLabel, posValue


# Main function ####
if __name__ == '__main__':
    """ .........
    To run the app, execute the following in terminal:

    [terminal_prompt]$ python addrNorm.py path/to/file.csv

    Currently the app supports images in the following formats: 
        .csv
    """
    RED='\033[0;31m'
    OKBLUE = '\033[3;94m'
    OKGREEN = '\033[1;92m'
    #OKGREEN = '\x1b[1;92m'
    NC='\033[0m' # No Color

    print(OKGREEN + "Init of Magic!" + NC)

    arg = sys.argv[1]
    data = jhon.read_csv(arg)

    #print(data)
    #print(len(data["DIRECCION"].unique()), type(data))

    addr = data["DIRECCION"]

    # Address schemes ####
    # Nucleo Urbano
    #city = ["CIUDAD"]
    upis = ["UPIS"]
    asen = ["AAHH", "AA HH", "A H", "A.A.H.H.", "ASENT H", "AH", "ASENTAMIENTO HUMANO", 
            "A  H", "AA  HH", "ASENT. H.", "ASENT.H.", "ASENT.HUMANO", 
            "ASENT. HUMANO", "AH.", "AA.HH"]
    pueJ = ["PPJJ", "PJ", "PJ.", "PP JJ", "PUEBLO JOVEN", "P.JOVEN", "P. JOVEN"]
    urba = ["URB.", "URBAN", "UR.", "URB", "URBANIZACION", "ENACE", "URBANIZ"]
    pueb = ["PUEBLO"]
    case = ["CASERIO"]
    anex = ["ANEXO"]
    coop = ["COOPERATIVA AGRARIA"]
    camp = ["CAMPAMENTO MINERO"]
    conj = ["CONJUNTO HABITACIONAL", "CONJ.HAB.", "CONJ HAB"]
    asoc = ["ASOCIACION", "ASOCIACIÓN", "ASOC"]
    cooV = ["COOPERATIVA DE VIVIENDA"]
    barr = ["BARRIO"]
    ccpp = ["CENTRO POBLADO", "CC.PP"]


    # TIPO DE VIA 
    call = ["CALLE", "CLL.", "CLL", "CAL", "CA"] 
    aven = ["AV.", "AVENIDA", "AVE", "AV"]
    jron = ["JR.", "JR", "JIRON"]
    carr = ["CARRETERA"]
    psje = ["PJE", "PSJE.", "PSJE", "PASAJE"]

    # NOMBRE DE VIA  
    # Número de puerta
    num = ["N°", " N ", "NRO", "NUMERO"]

    # BLOCK 
    blck = ["BLOCK", "BL"] 

    # PISO  
    piso = ["PISO"]

    # INTERIOR   
    inte = ["INT.", "INTE", "INT", "DPT.", "DPTO.", "DEPA", "DPT", "DPTO"]

    # MANZANA 
    mzna = ["MZA", "MZ.", "MZ", "MAZ", "MANZANA", "MANZ"]

    # LOTE 
    lote = ["LT", "LT.", "LOTE", "LTE", "LOT"]

    # KILOMETRO

    
    # Pos of address identifiers ####
    indAddr = addr[218]
    print("Address: ", indAddr)

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

    print(sorted(addrStructure, key=addrStructure.get))

    for val in addrStructure:
        print(val, addrStructure[val])
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