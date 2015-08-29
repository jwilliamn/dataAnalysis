# Clasificación de archivos por Hogar - Individuos & Bienes

#library(foreign)  # library to read .dta
#library(haven)
#library(memisc)

#modulo <- read.dta("enaho01_2013_200.dta", convert.factors = FALSE)
#write.table(modulo, file = "modulo.csv", sep = ",", row.names = F)
#modulo <- read.csv("modulo.csv")


# Primero configuramos el directorio de trabajo
wd <- "/home/williamn/Repository/bitBucket/midis"    # ruta del directorio de trabajo
setwd(wd)
dPath <- "/home/williamn/Repository/data/midis"    # ruta general que contiene data
sdir <- "data.2007"    # directorio fuente que contiene la data
ddir <- "data_clasificada_2007"  # directorio destino donde se guarda la data clasificada
# verifica si existe el directorio destino
if(!dir.exists(file.path(dPath, ddir))){ dir.create(file.path(dPath, ddir))}

if(!dir.exists(paste(dPath, ddir, "individual", sep = "/"))){  
    dir.create(paste(dPath, ddir, "individual", sep = "/"))}
if(!dir.exists(paste(dPath, ddir, "bienes", sep = "/"))){  
    dir.create(paste(dPath, ddir, "bienes", sep = "/"))}
if(!dir.exists(paste(dPath, ddir, "hogar", sep = "/"))){  
    dir.create(paste(dPath, ddir, "hogar", sep = "/"))}


# Clasificamos los archivos de acuerdo a la llave (codperso, p612n, hogar)
listFiles <- list.files(file.path(dPath, sdir), full.names = T)
nameFiles <- list.files(file.path(dPath, sdir))

individual <- data.frame()
bienes <- data.frame()
hogar <- data.frame()
for(i in 1:length(listFiles)){
    tmp <- read.csv(listFiles[i], header = TRUE)
    sname <- listFiles[i]
    name <- nameFiles[i]
    flag <- FALSE
    for(j in 1:13){
        if(colnames(tmp)[j] == "codperso"){ 
            individual <- rbind(individual, cbind(dim(tmp)[1], dim(tmp)[2], name))
            file.copy(from = paste(sname, sep = "/"), 
                      to = paste(dPath, ddir, "individual", name, sep = "/"))
            flag <- TRUE
        }
        if(colnames(tmp)[j] == "p612n"){ 
            bienes <- rbind(bienes, cbind(dim(tmp)[1], dim(tmp)[2], name))
            file.copy(from = paste(sname, sep = "/"), 
                      to = paste(dPath, ddir, "bienes", name, sep = "/"))
            flag <- TRUE
        }
    }
    if(flag==FALSE){
        hogar <- rbind(hogar, cbind(dim(tmp)[1], dim(tmp)[2], name))
        file.copy(from = paste(sname, sep = "/"), 
                  to = paste(dPath, ddir, "hogar", name, sep = "/"))
    }
}
colnames(individual) <- c("Obs.", "Caract.", "Módulo")
colnames(bienes) <- c("Obs.", "Caract.", "Módulo")
colnames(hogar) <- c("Obs.", "Caract.", "Módulo")

write.table(individual, file = paste(dPath, ddir, "individual.txt", sep = "/"), 
            sep = ",", row.names = F, quote = F)
write.table(bienes, file = paste(dPath, ddir, "bienes.txt", sep = "/"), 
            sep = ",", row.names = F, quote = F)
write.table(hogar, file = paste(dPath, ddir, "hogar.txt", sep = "/"), 
            sep = ",", row.names = F, quote = F)


# Analizamos la estructura de los datos

## Indicamos el subdirectorio donde se encuentran la data clasificada
directories <- (c("data_clasificada_2007", "data_clasificada_2008", 
                  "data_clasificada_2009", "data_clasificada_2010", 
                  "data_clasificada_2011", "data_clasificada_2012", 
                  "data_clasificada_2013", "data_clasificada_2014"))

## Función para obtener un substring cualquiera
anySubstr <- function(x, ni, nf){
    substr(x, ni, nchar(x)-nf-1)
}

## Módulo base(2007) que se compararan con los módulos de otros años.
tmpPath <- file.path(dPath, directories[1])
tmpidir <- "individual"

tmpFile <- list.files(file.path(tmpPath, tmpidir), full.names = T)
tmpModulo <- read.csv(tmpFile[1])    # módulo base
tmpNFeat <- dim(tmpModulo)[2]    # número de características base
tmpFeat <- colnames(tmpModulo)    # características del módulo base

tmpNModulo <- character(0) # nombre de todos los módulos para el año base
for(j in 1:length(tmpFile)){
    tmpNModulo <- c(tmpNModulo, anySubstr(tmpFile[j], 59, 3))}

## Empezamos a leer y comparar los módulos a partir del 2008 con el módulo base
for(i in 2:length(directories)){
    newdPath <- file.path(dPath, directories[i])
    idir <- "individual"    # directorio de archivos clasificados como Individual
    
    indFiles <- list.files(file.path(newdPath, idir), full.names = T)
    cat("\n\nResumen de estructura de datos para los módulos del año: ", 
        anySubstr(directories[i], 18, -1))
    
    nmodulo <- character(0) # nombre de todos los módulos para el año indicado
    for(j in 1:length(indFiles)){
        nmodulo <- c(nmodulo, anySubstr(indFiles[j], 59, 3))}
    
    ### Comparamos que módulos se agregaron respecto al año anterior
    varModulo <- setdiff(nmodulo, tmpNModulo)
    cat("\n\nNuevos módulos que son diferentes respecto al año anterior:\n[", varModulo, "]")
    tmpNModulo <- nmodulo
    
    ### Se comparan con las reglas encontradas en el módulo base
    numCumple <- c(0)  # número de variables que cumple con las reglas
    if("Modulo02" %in% nmodulo){
        if(i == 2){
            tmpModulo02 <- read.csv(tmpFile[1])    # módulo base
            tmpNFeat02 <- dim(tmpModulo02)[2]    # número de características base
            tmpFeat02 <- colnames(tmpModulo02)    # características del módulo base
            rm(tmpModulo02)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo02")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo02")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat02    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat02, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat02)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        if("p205" %in% same){
            if(sum(is.na(modulo$p205)) == (sum(is.na(modulo$p204)) + length(which(modulo$p204==2)))){
                numCumple <- numCumple + 1
                cat("\n-- p205 Ok")
            }else{cat("\n-- p205 No cumple")}
        }else{cat("\n-- p205 No existe en este módulo!")}
        
        if("p206" %in% same){
            if(sum(is.na(modulo$p206)) == (sum(is.na(modulo$p204)) + length(which(modulo$p204==1)))){
                numCumple <- numCumple + 1
                cat("\n-- p206 Ok")
            }else{cat("\n-- p206 No cumple")}
        }else{cat("\n-- p206 No existe en este módulo!")}
        
        if("p208b" %in% same){
            if(sum(is.na(modulo$p208b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p208b Ok (eliminar)")
            }else{cat("\n-- p208b No cumple")}
        }else{cat("\n-- p208b No existe en este módulo!")}
        
        if("p208a2" %in% same){
            if(sum(is.na(modulo$p208a2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p208a2 Ok (distritos sin código)")
            }else{cat("\n-- p208a2 No cumple")}
        }else{cat("\n-- p208a2 No existe en este módulo!")}
        
        if("p209" %in% same){
            if(sum(is.na(modulo$p209)) == length(which(modulo$p208a < 12)) + sum(is.na(modulo$p208a))){
                numCumple <- numCumple + 1
                cat("\n-- p209 Ok")
            }else{cat("\n-- p209 No cumple")}
        }else{cat("\n-- p209 No existe en este módulo!")}
        
        if("p210" %in% same){
            if(sum(is.na(modulo$p210)) == (length(which(modulo$p208a < 6)) + 
                                           length(which(modulo$p208a > 20)) + sum(is.na(modulo$p208a)))){
                numCumple <- numCumple + 1
                cat("\n-- p210 Ok")
            }else{cat("\n-- p210 No cumple")}
        }else{cat("\n-- p210 No existe en este módulo!")}
        
        if("p211" %in% same){
            if(sum(is.na(modulo$p211)) >= (length(which(modulo$p208a < 6)) + 
                                           length(which(modulo$p208a > 20)) + sum(is.na(modulo$p208a)))){
                numCumple <- numCumple + 1
                cat("\n-- p211 Ok")
            }else{cat("\n-- p211 No cumple")}
        }else{cat("\n-- p211 No existe en este módulo!")}
        
        if("p212" %in% same){
            if(sum(is.na(modulo$p212)) >= length(which(modulo$p208a < 3)) + sum(is.na(modulo$p208a))){
                numCumple <- numCumple + 1
                cat("\n-- p212 Ok")
            }else{cat("\n-- p212 No cumple")}
        }else{cat("\n-- p212 No existe en este módulo!")}
        
        if("p213" %in% same){
            if(sum(is.na(modulo$p213)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p213 Ok")
            }else{cat("\n-- p213 No cumple")}
        }else{cat("\n-- p213 No existe en este módulo!")}
        
        if("p214" %in% same){
            if(sum(is.na(modulo$p214)) >= length(which(modulo$p208a < 14)) + sum(is.na(modulo$p208a))){
                numCumple <- numCumple + 1
                cat("\n-- p214 Ok")
            }else{cat("\n-- p214 No cumple")}
        }else{cat("\n-- p214 No existe en este módulo!")}
        
        if("t211" %in% same){
            if(sum(is.na(modulo$t211)) >= (length(which(modulo$p208a < 6)) + 
                                           length(which(modulo$p208a > 20)) + sum(is.na(modulo$p208a)))){
                numCumple <- numCumple + 1
                cat("\n-- t211 Ok")
            }else{cat("\n-- t211 No cumple")}
        }else{cat("\n-- t211 No existe en este módulo!")}
        
        tmpNFeat02 <- nfeat
        tmpFeat02 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo03" %in% nmodulo){
        if(i == 2){
            tmpModulo03 <- read.csv(tmpFile[2])    # módulo base
            tmpNFeat03 <- dim(tmpModulo03)[2]    # número de características base
            tmpFeat03 <- colnames(tmpModulo03)    # características del módulo base
            rm(tmpModulo03)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo03")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo03")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat03    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat03, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat03)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        if("p300a" %in% same){
            if(sum(is.na(modulo$p300a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p300a Ok")
            }else{cat("\n-- p300a No cumple")}
        }else{cat("\n-- p300a No existe en este módulo!")}
        
        if("p301a" %in% same){
            if(sum(is.na(modulo$p301a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p301a Ok")
            }else{cat("\n-- p301a No cumple")}
        }else{cat("\n-- p301a No existe en este módulo!")}
        
        if("p301b" %in% same){
            if(sum(is.na(modulo$p301b)) >= length(which(modulo$p301a==1))){
                numCumple <- numCumple + 1
                cat("\n-- p301b Ok")
            }else{cat("\n-- p301b No cumple")}
        }else{cat("\n-- p301b No existe en este módulo!")}
        
        if("p301c" %in% same){
            if(sum(is.na(modulo$p301c)) >= length(which(modulo$p301a==1))){
                numCumple <- numCumple + 1
                cat("\n-- p301c Ok")
            }else{cat("\n-- p301c No cumple")}
        }else{cat("\n-- p301c No existe en este módulo!")}
        
        if("p301d" %in% same){
            if(sum(is.na(modulo$p301d)) >= length(which(modulo$p301a==1))){
                numCumple <- numCumple + 1
                cat("\n-- p301d Ok")
            }else{cat("\n-- p301d No cumple")}
        }else{cat("\n-- p301d No existe en este módulo!")}
        
        if("p301a0" %in% same){
            if(sum(is.na(modulo$p301a0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p301a0 Ok")
            }else{cat("\n-- p301a0 No cumple")}
        }else{cat("\n-- p301a0 No existe en este módulo!")}
        
        if("p301a1" %in% same){
            if(sum(is.na(modulo$p301a1)) >= sum(is.na(modulo$p301a0))){
                numCumple <- numCumple + 1
                cat("\n-- p301a1 Ok")
            }else{cat("\n-- p301a1 No cumple")}
        }else{cat("\n-- p301a1 No existe en este módulo!")}
        
        if("p302" %in% same){
            if(sum(is.na(modulo$p302)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p302 Ok")
            }else{cat("\n-- p302 No cumple")}
        }else{cat("\n-- p302 No existe en este módulo!")}
        
        if("p302x" %in% same){
            if(sum(is.na(modulo$p302x)) >= sum(is.na(modulo$p302))){
                numCumple <- numCumple + 1
                cat("\n-- p302x Ok")
            }else{cat("\n-- p302x No cumple")}
        }else{cat("\n-- p302x No existe en este módulo!")}
        
        if("p302a" %in% same){
            if(sum(is.na(modulo$p302a)) == sum(is.na(modulo$p302))){
                numCumple <- numCumple + 1
                cat("\n-- p302a Ok")
            }else{cat("\n-- p302a No cumple")}
        }else{cat("\n-- p302a No existe en este módulo!")}
        
        if("p302b" %in% same){
            if(sum(is.na(modulo$p302b)) == sum(is.na(modulo$p302a))){
                numCumple <- numCumple + 1
                cat("\n-- p302b Ok")
            }else{cat("\n-- p302b No cumple")}
        }else{cat("\n-- p302b No existe en este módulo!")}
        
        if("p303" %in% same){
            if(sum(is.na(modulo$p303)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p303 Ok")
            }else{cat("\n-- p303 No cumple")}
        }else{cat("\n-- p303 No existe en este módulo!")}
        
        if("p304a" %in% same){
            if(sum(is.na(modulo$p304a)) >= length(which(modulo$p303==2))){
                numCumple <- numCumple + 1
                cat("\n-- p304a Ok")
            }else{cat("\n-- p304a No cumple")}
        }else{cat("\n-- p304a No existe en este módulo!")}
        
        if("p304b" %in% same){
            if(sum(is.na(modulo$p304b)) >= sum(is.na(modulo$p304a))){
                numCumple <- numCumple + 1
                cat("\n-- p304b Ok")
            }else{cat("\n-- p304b No cumple")}
        }else{cat("\n-- p304b No existe en este módulo!")}
        
        if("p304c" %in% same){
            if(sum(is.na(modulo$p304c)) >= sum(is.na(modulo$p304a))){
                numCumple <- numCumple + 1
                cat("\n-- p304c Ok")
            }else{cat("\n-- p304c No cumple")}
        }else{cat("\n-- p304c No existe en este módulo!")}
        
        if("p304d" %in% same){
            if(sum(is.na(modulo$p304d)) >= sum(is.na(modulo$p304a))){
                numCumple <- numCumple + 1
                cat("\n-- p304d Ok")
            }else{cat("\n-- p304d No cumple")}
        }else{cat("\n-- p304d No existe en este módulo!")}
        
        if("p305" %in% same){
            if(sum(is.na(modulo$p305)) >= sum(is.na(modulo$p304a))){
                numCumple <- numCumple + 1
                cat("\n-- p305 Ok")
            }else{cat("\n-- p305 No cumple")}
        }else{cat("\n-- p305 No existe en este módulo!")}
        
        if("p306" %in% same){
            if(sum(is.na(modulo$p306)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p306 Ok")
            }else{cat("\n-- p306 No cumple")}
        }else{cat("\n-- p306 No existe en este módulo!")}
        
        if("p307" %in% same){
            if(sum(is.na(modulo$p307)) >= length(which(modulo$p306==2)) + sum(is.na(modulo$p306))){
                numCumple <- numCumple + 1
                cat("\n-- p307 Ok")
            }else{cat("\n-- p307 No cumple")}
        }else{cat("\n-- p307 No existe en este módulo!")}
        
        if("p308a" %in% same){
            if(sum(is.na(modulo$p308a)) >= length(which(modulo$p306==2)) + sum(is.na(modulo$p306))){
                numCumple <- numCumple + 1
                cat("\n-- p308a Ok")
            }else{cat("\n-- p308a No cumple")}
        }else{cat("\n-- p308a No existe en este módulo!")}
        
        if("p308b" %in% same){
            if(sum(is.na(modulo$p308b)) >= sum(is.na(modulo$p308a))){
                numCumple <- numCumple + 1
                cat("\n-- p308b Ok")
            }else{cat("\n-- p308b No cumple")}
        }else{cat("\n-- p308b No existe en este módulo!")}
        
        if("p308c" %in% same){
            if(sum(is.na(modulo$p308c)) >= sum(is.na(modulo$p308b))){
                numCumple <- numCumple + 1
                cat("\n-- p308c Ok")
            }else{cat("\n-- p308c No cumple")}
        }else{cat("\n-- p308c No existe en este módulo!")}
        
        if("p308d" %in% same){
            if(sum(is.na(modulo$p308d)) == sum(is.na(modulo$p308a))){
                numCumple <- numCumple + 1
                cat("\n-- p308d Ok")
            }else{cat("\n-- p308d No cumple")}
        }else{cat("\n-- p308d No existe en este módulo!")}
        
        if("p3091a" %in% same){
            if(sum(is.na(modulo$p3091a)) <= sum(is.na(modulo$p308a))){
                numCumple <- numCumple + 1
                cat("\n-- p3091a Ok (p308a tiene mas NA's que p3091a, 
                    cuando debería ser al revez)")
            }else{cat("\n-- p3091a No cumple")}
            }else{cat("\n-- p3091a No existe en este módulo!")}
        
        if("p3091b" %in% same){
            if(sum(is.na(modulo$p3091b)) >= (length(which(modulo$p3091a==2)) + 
                                             length(which(modulo$p3091a==3)) + 
                                             sum(is.na(modulo$p3091a)))){
                numCumple <- numCumple + 1
                cat("\n-- p3091b Ok")
            }else{cat("\n-- p3091b No cumple")}
        }else{cat("\n-- p3091b No existe en este módulo!")}
        
        if("p3091c" %in% same){
            if(sum(is.na(modulo$p3091c)) == sum(is.na(modulo$p3091b))){
                numCumple <- numCumple + 1
                cat("\n-- p3091c Ok")
            }else{cat("\n-- p3091c No cumple")}
        }else{cat("\n-- p3091c No existe en este módulo!")}
        
        if("p310" %in% same){
            if(sum(!is.na(modulo$p310)) <= (length(which(modulo$p306==2)) + 
                                            length(which(modulo$p307==2)) + 
                                            length(which(modulo$p3091a==2)) + 
                                            length(which(modulo$p3092a==2)))){
                numCumple <- numCumple + 1
                cat("\n-- p310 Ok")
            }else{cat("\n-- p310 No cumple")}
        }else{cat("\n-- p310 No existe en este módulo!")}
        
        if("p311n_1" %in% same){
            if(sum(is.na(modulo$p311n_1)) == (length(which(modulo$p306==2)) + 
                                              sum(is.na(modulo$p306)))){
                numCumple <- numCumple + 1
                cat("\n-- p311n_1 Ok")
            }else{cat("\n-- p311n_1 No cumple")}
        }else{cat("\n-- p311n_1 No existe en este módulo!")}
        
        if("p311_1" %in% same){
            if(sum(is.na(modulo$p311_1)) == (length(which(modulo$p306==2)) + 
                                             sum(is.na(modulo$p306)))){
                numCumple <- numCumple + 1
                cat("\n-- p311_1 Ok")
            }else{cat("\n-- p311_1 No cumple")}
        }else{cat("\n-- p311_1 No existe en este módulo!")}
        
        if("p311a1_1" %in% same){
            if(sum(is.na(modulo$p311a1_1)) >= (length(which(modulo$p310==2)) + 
                                               sum(is.na(modulo$p310)))){
                numCumple <- numCumple + 1
                cat("\n-- p311a1_1 Ok")
            }else{cat("\n-- p311a1_1 No cumple")}
        }else{cat("\n-- p311a1_1 No existe en este módulo!")}
        
        if("p311b_1" %in% same){
            if(sum(is.na(modulo$p311b_1)) >= sum(is.na(modulo$p311a1_1))){
                numCumple <- numCumple + 1
                cat("\n-- p311b_1 Ok")
            }else{cat("\n-- p311b_1 No cumple")}
        }else{cat("\n-- p311b_1 No existe en este módulo!")}
        
        if("p311c_1" %in% same){
            if(sum(is.na(modulo$p311c_1)) <= sum(is.na(modulo$p311b_1))){
                numCumple <- numCumple + 1
                cat("\n-- p311c_1 Ok")
            }else{cat("\n-- p311c_1 No cumple")}
        }else{cat("\n-- p311c_1 No existe en este módulo!")}
        
        if("p311d_1" %in% same){
            if(sum(is.na(modulo$p311d_1)) == (length(which(modulo$p311a2_1==0)) + 
                                              sum(is.na(modulo$p311a2_1)))){
                numCumple <- numCumple + 1
                cat("\n-- p311d_1 Ok")
            }else{cat("\n-- p311d_1 No cumple (esta variable no estaba muy clara)")}
        }else{cat("\n-- p311d_1 No existe en este módulo!")}
        
        if("p311e_1" %in% same){
            if(sum(is.na(modulo$p311e_1)) >= sum(is.na(modulo$p311a1_1))){
                numCumple <- numCumple + 1
                cat("\n-- p311e_1 Ok")
            }else{cat("\n-- p311e_1 No cumple")}
        }else{cat("\n-- p311e_1 No existe en este módulo!")}
        
        if("p311t1" %in% same){
            if(sum(is.na(modulo$p311t1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p311t1 Ok")
            }else{cat("\n-- p311t1 No cumple")}
        }else{cat("\n-- p311t1 No existe en este módulo!")}
        
        if("p311t2" %in% same){
            if(sum(is.na(modulo$p311t2)) == sum(is.na(modulo$p311t1))){
                numCumple <- numCumple + 1
                cat("\n-- p311t2 Ok")
            }else{cat("\n-- p311t2 No cumple")}
        }else{cat("\n-- p311t2 No existe en este módulo!")}
        
        if("p3121" %in% same){
            if(sum(is.na(modulo$p3121)) >= sum(is.na(modulo$p306))){
                numCumple <- numCumple + 1
                cat("\n-- p3121 Ok")
            }else{cat("\n-- p3121 No cumple")}
        }else{cat("\n-- p3121 No existe en este módulo!")}
        
        if("p3121a1" %in% same){
            if(sum(is.na(modulo$p3121a1)) == (length(which(modulo$p3121==2)) + 
                                              sum(is.na(modulo$p3121)))){
                numCumple <- numCumple + 1
                cat("\n-- p3121a1 Ok")
            }else{cat("\n-- p3121a1 No cumple")}
        }else{cat("\n-- p3121a1 No existe en este módulo!")}
        
        if("p3121b" %in% same){
            if(sum(is.na(modulo$p3121b)) >= sum(is.na(modulo$p3121a1))){
                numCumple <- numCumple + 1
                cat("\n-- p3121b Ok")
            }else{cat("\n-- p3121b No cumple")}
        }else{cat("\n-- p3121b No existe en este módulo!")}
        
        if("p3122" %in% same){
            if(sum(is.na(modulo$p3122)) >= sum(is.na(modulo$p306))){
                numCumple <- numCumple + 1
                cat("\n-- p3122 Ok")
            }else{cat("\n-- p3122 No cumple")}
        }else{cat("\n-- p3122 No existe en este módulo!")}
        
        if("p3122a1" %in% same){
            if(sum(is.na(modulo$p3122a1)) == (length(which(modulo$p3122==2)) + 
                                              sum(is.na(modulo$p3122)))){
                numCumple <- numCumple + 1
                cat("\n-- p3122a1 Ok")
            }else{cat("\n-- p3122a1 No cumple")}
        }else{cat("\n-- p3122a1 No existe en este módulo!")}
        
        if("p3122b" %in% same){
            if(sum(is.na(modulo$p3122b)) >= sum(is.na(modulo$p3122a1))){
                numCumple <- numCumple + 1
                cat("\n-- p3122b Ok")
            }else{cat("\n-- p3122b No cumple")}
        }else{cat("\n-- p3122b No existe en este módulo!")}
        
        if("p312t1" %in% same){
            if(sum(is.na(modulo$p312t1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p312t1 Ok")
            }else{cat("\n-- p312t1 No cumple")}
        }else{cat("\n-- p312t1 No existe en este módulo!")}
        
        if("p312t2" %in% same){
            if(sum(is.na(modulo$p312t2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p312t2 Ok")
            }else{cat("\n-- p312t2 No cumple")}
        }else{cat("\n-- p312t2 No existe en este módulo!")}
        
        if("p313a" %in% same){
            if(sum(is.na(modulo$p313a)) >= sum(is.na(modulo$p307))){
                numCumple <- numCumple + 1
                cat("\n-- p313a Ok")
            }else{cat("\n-- p313a No cumple")}
        }else{cat("\n-- p313a No existe en este módulo!")}
        
        if("p314a" %in% same){
            if(sum(is.na(modulo$p314a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- p314a Ok")
            }else{cat("\n-- p314a No cumple")}
        }else{cat("\n-- p314a No existe en este módulo!")}
        
        if("p314b_1" %in% same){
            if(sum(is.na(modulo$p314b_1)) == (length(which(modulo$p314a==2)) + 
                                              sum(is.na(modulo$p314a)))){
                numCumple <- numCumple + 1
                cat("\n-- p314b_1 Ok")
            }else{cat("\n-- p314b_1 No cumple")}
        }else{cat("\n-- p314b_1 No existe en este módulo!")}
        
        if("p314c" %in% same){
            if(sum(is.na(modulo$p314c)) >= (length(which(modulo$p314a==2)) + 
                                            sum(is.na(modulo$p314a)))){
                numCumple <- numCumple + 1
                cat("\n-- p314c Ok")
            }else{cat("\n-- p314c No cumple")}
        }else{cat("\n-- p314c No existe en este módulo!")}
        
        if("p314d" %in% same){
            if(sum(is.na(modulo$p314d)) >= sum(is.na(modulo$p314c))){
                numCumple <- numCumple + 1
                cat("\n-- p314d Ok")
            }else{cat("\n-- p314d No cumple")}
        }else{cat("\n-- p314d No existe en este módulo!")}
        
        if("p3151" %in% same){
            if(sum(is.na(modulo$p3151)) >= sum(is.na(modulo$p314b_1))){
                numCumple <- numCumple + 1
                cat("\n-- p3151 Ok")
            }else{cat("\n-- p3151 No cumple")}
        }else{cat("\n-- p3151 No existe en este módulo!")}
        
        if("p315a" %in% same){
            if(sum(is.na(modulo$p315a)) >= sum(is.na(modulo$p3151))){
                numCumple <- numCumple + 1
                cat("\n-- p315a Ok")
            }else{cat("\n-- p315a No cumple")}
        }else{cat("\n-- p315a No existe en este módulo!")}
        
        if("p315b" %in% same){
            if(sum(is.na(modulo$p315b)) >= sum(is.na(modulo$p3151))){
                numCumple <- numCumple + 1
                cat("\n-- p315b Ok")
            }else{cat("\n-- p315b No cumple")}
        }else{cat("\n-- p315b No existe en este módulo!")}
        
        if("p316_1" %in% same){
            if(sum(is.na(modulo$p316_1)) >= (length(which(modulo$p314a==2)) + 
                                             sum(is.na(modulo$p314a)))){
                numCumple <- numCumple + 1
                cat("\n-- p316_1 Ok")
            }else{cat("\n-- p316_1 No cumple")}
        }else{cat("\n-- p316_1 No existe en este módulo!")}
        
        if("p316a_1" %in% same){
            if(sum(is.na(modulo$p316a_1)) >= (length(which(modulo$p316_1==2)) + 
                                              sum(is.na(modulo$p316_1)))){
                numCumple <- numCumple + 1
                cat("\n-- p316a_1 Ok")
            }else{cat("\n-- p316a_1 No cumple")}
        }else{cat("\n-- p316a_1 No existe en este módulo!")}
        
        if("t313a" %in% same){
            if(sum(is.na(modulo$t313a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n-- t313a Ok")
            }else{cat("\n-- t313a No cumple")}
        }else{cat("\n-- t313a No existe en este módulo!")}
        
        tmpNFeat03 <- nfeat
        tmpFeat03 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
        }
    
    if("Modulo04" %in% nmodulo){
        if(i == 2){
            tmpModulo04 <- read.csv(tmpFile[3])    # módulo base
            tmpNFeat04 <- dim(tmpModulo04)[2]    # número de características base
            tmpFeat04 <- colnames(tmpModulo04)    # características del módulo base
            rm(tmpModulo04)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo04")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo04")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat04    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat04, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat04)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p417_01"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_01)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_03"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_03)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_04"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_04)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_05"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_05)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_06"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_06)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_07"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_07)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_09"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_09)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_10"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_10)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_13"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_13)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_14"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_14)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_15"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_15)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p400a1"
        if(colnm %in% same){
            if(sum(is.na(modulo$p400a1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p400a2"
        if(colnm %in% same){
            if(sum(is.na(modulo$p400a2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p400a3"
        if(colnm %in% same){
            if(sum(is.na(modulo$p400a3)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p401"
        if(colnm %in% same){
            if(sum(is.na(modulo$p401)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p401a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p401a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p401a1"
        if(colnm %in% same){
            if(sum(is.na(modulo$p401a1)) >= (length(which(modulo$p401a==2)) + 
                                             sum(is.na(modulo$p401a)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p401b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p401b)) == (length(which(modulo$p401a==2)) + 
                                            sum(is.na(modulo$p401a)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4021"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4021)) == sum(is.na(modulo$p401))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4021n"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4021n)) == (length(which(modulo$p4021==0)) + 
                                             sum(is.na(modulo$p4021)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4022n"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4022n)) == (length(which(modulo$p4022==0)) + 
                                             sum(is.na(modulo$p4022)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4023n"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4023n)) == (length(which(modulo$p4023==0)) + 
                                             sum(is.na(modulo$p4023)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4024n"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4024n)) == (length(which(modulo$p4024==0)) + 
                                             sum(is.na(modulo$p4024)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4031"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4031)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok (no se encontró relación clara)")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4041"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4041)) >= sum(is.na(modulo$p4031))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4061"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4061)) >= sum(is.na(modulo$p4031))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4061a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4061a)) >= sum(is.na(modulo$p4061))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p407a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p407a)) >= sum(is.na(modulo$p4061))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4091"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4091)) >= sum(is.na(modulo$p4031))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p410a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p410a)) == sum(is.na(modulo$p4021))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p410b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p410b)) == (length(which(modulo$p410a==2)) + 
                                            sum(is.na(modulo$p410a)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4111"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4111)) == (length(which(modulo$p410a==2)) + 
                                            sum(is.na(modulo$p410a)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p412"
        if(colnm %in% same){
            if(sum(is.na(modulo$p412)) == sum(is.na(modulo$p410a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4131"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4131)) == sum(is.na(modulo$p410a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4131a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4131a)) == (length(which(modulo$p4131==3)) + 
                                             length(which(modulo$p4131==2)) + 
                                             sum(is.na(modulo$p4131)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p414n_01"
        if(colnm %in% same){
            if(sum(is.na(modulo$p414n_01)) + length(which(modulo$p414n_01==1)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok (solo toma el valor de 1's)")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p414_01"
        if(colnm %in% same){
            if(sum(is.na(modulo$p414_01)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4151_01"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4151_01)) == (length(which(modulo$p414_01==2)) + 
                                               sum(is.na(modulo$p414_01)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4151001"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4151001)) + length(which(modulo$p4151001==0)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p41601"
        if(colnm %in% same){
            if(sum(is.na(modulo$p41601)) >= (length(which(modulo$p414_01==2)) + 
                                             sum(is.na(modulo$p414_01)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_02"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_02)) >= sum(is.na(modulo$p414_01))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_08"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_08)) >= sum(is.na(modulo$p41608))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_11"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_11)) <= sum(is.na(modulo$p41611))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p417_12"
        if(colnm %in% same){
            if(sum(is.na(modulo$p417_12)) >= sum(is.na(modulo$p41612))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p41801"
        if(colnm %in% same){
            if(sum(is.na(modulo$p41801)) >= sum(is.na(modulo$p4151_01))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4191"
        if(colnm %in% same){
            if(sum(is.na(modulo$p4191)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "blibre07"
        if(colnm %in% same){
            if(sum(is.na(modulo$blibre07)) + length(which(modulo$blibre07==1)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat04 <- nfeat
        tmpFeat04 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo05" %in% nmodulo){
        if(i == 2){
            tmpModulo05 <- read.csv(tmpFile[4])    # módulo base
            tmpNFeat05 <- dim(tmpModulo05)[2]    # número de características base
            tmpFeat05 <- colnames(tmpModulo05)    # características del módulo base
            rm(tmpModulo05)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo05")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo05")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat05    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat05, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat05)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p5405b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p5405b)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5405c"
        if(colnm %in% same){
            if(sum(is.na(modulo$p5405c)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p500b1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p500b1)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p500d1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p500d1)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p500n"
        if(colnm %in% same){
            eq <- modulo$codperso == modulo$p500n
            if(!"FALSE" %in% eq ==TRUE){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p500i"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p500i)) == dim(modulo)[1]){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok (eliminar esta variable)")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p501"
        if(colnm %in% same){
            if(sum(is.na(modulo$p501)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p502"
        if(colnm %in% same){
            if(sum(is.na(modulo$p502)) == (length(which(modulo$p501==1)) + 
                                           sum(is.na(modulo$p501)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p503"
        if(colnm %in% same){
            if(sum(is.na(modulo$p503)) == (length(which(modulo$p502==1)) + 
                                           sum(is.na(modulo$p502)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p504"
        if(colnm %in% same){
            if(sum(is.na(modulo$p504)) == (length(which(modulo$p503==1)) + 
                                           sum(is.na(modulo$p503)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5041"
        if(colnm %in% same){
            if(sum(is.na(modulo$p5041)) == sum(is.na(modulo$p504))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p505"
        if(colnm %in% same){
            if(sum(is.na(modulo$p505)) == (length(which(modulo$p504==0)) + 
                                           sum(is.na(modulo$p504)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p505b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p505b)) >= sum(is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p506"
        if(colnm %in% same){
            if(sum(is.na(modulo$p506)) >= sum(is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p506r4"
        if(colnm %in% same){
            if(sum(is.na(modulo$p506r4)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok (en el 2007 no aparece en el diccionario)")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p507"
        if(colnm %in% same){
            if(sum(is.na(modulo$p507)) == sum(is.na(modulo$p505b))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p508"
        if(colnm %in% same){
            if(sum(is.na(modulo$p508)) >= sum(is.na(modulo$p507))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p509"
        if(colnm %in% same){
            if(sum(is.na(modulo$p509)) == sum(is.na(modulo$p508))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok (No es muy clara la relación)")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p510"
        if(colnm %in% same){
            if(sum(is.na(modulo$p510)) >= sum(is.na(modulo$p507))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p510a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p510a)) >= sum(is.na(modulo$p506))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p510b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p510b)) == sum(is.na(modulo$p510a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5111"
        if(colnm %in% same){
            if(sum(is.na(modulo$p5111)) >= sum(is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p511a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p511a)) == (length(which(modulo$p510==2)) + 
                                            length(which(modulo$p510==3)) + 
                                            length(which(modulo$p510==4)) + 
                                            length(which(modulo$p510==5)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p512a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p512a)) >= sum(is.na(modulo$p5111))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p512b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p512b)) >= sum(is.na(modulo$p512a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p513"
        if(colnm %in% same){
            if(sum(is.na(modulo$p513)) >= sum(is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p513a1"
        if(colnm %in% same){
            if(sum(is.na(modulo$p513a1)) >= sum(is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p514"
        if(colnm %in% same){
            if(sum(is.na(modulo$p514)) >= sum(is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5151"
        if(colnm %in% same){
            if(sum(is.na(modulo$p5151)) == (length(which(modulo$p514==1)) + 
                                            sum(is.na(modulo$p514)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p516"
        if(colnm %in% same){
            if(sum(is.na(modulo$p516)) == (length(which(modulo$p514==2)) + 
                                           sum(is.na(modulo$p514)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p516r4x"
        if(colnm %in% same){
            if(sum(is.na(modulo$p516r4x)) == sum(is.na(modulo$p516))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p517"
        if(colnm %in% same){
            if(sum(is.na(modulo$p517)) == (length(which(modulo$p514==2)) + 
                                           sum(is.na(modulo$p514)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p517a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p517a)) >= (length(which(modulo$p517==3)) + 
                                            length(which(modulo$p517==4)) + 
                                            length(which(modulo$p517==7)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p517b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p517b)) >= (length(which(modulo$p517a==6)) + 
                                            length(which(modulo$p517a==7)) + 
                                            length(which(modulo$p517==1)) + 
                                            length(which(modulo$p517==2)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p517d1"
        if(colnm %in% same){
            if(sum(is.na(modulo$p517d1)) >= (length(which(modulo$p517==5)) + 
                                             length(which(modulo$p517==6)) + 
                                             length(which(modulo$p517a==2)) + 
                                             length(which(modulo$p517a==3)) + 
                                             length(which(modulo$p517a==4)) + 
                                             length(which(modulo$p517a==5)) + 
                                             sum(is.na(modulo$p517b)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p517d2"
        if(colnm %in% same){
            if(sum(is.na(modulo$p517d2)) >= sum(is.na(modulo$p517d1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p518"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p518)) <= sum(!is.na(modulo$p517d1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p519"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p519)) <= sum(!is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p520"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p520)) <= sum(!is.na(modulo$p519))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p521"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p521)) <= sum(!is.na(modulo$p519))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p521a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p521a)) <= sum(!is.na(modulo$p521))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p523"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p523)) <= (length(which(modulo$p507==3)) + 
                                            length(which(modulo$p507==4)) + 
                                            length(which(modulo$p507==6)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p524a1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p524a1)) <= sum(!is.na(modulo$p523))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p524a2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p524a2)) <= sum(!is.na(modulo$p523))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p528"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p528)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5291a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5291a)) <= sum(!is.na(modulo$p528))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5291b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5291b)) <= sum(!is.na(modulo$p5291a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5291c"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5291c)) <= sum(!is.na(modulo$p5291a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p529t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p529t)) <= sum(!is.na(modulo$p528))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5297a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5297a)) <= sum(!is.na(modulo$p528))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p530a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p530a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p530b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p530b)) >= sum(!is.na(modulo$p530a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p535"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p535)) <= sum(!is.na(modulo$p530b))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p536"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p536)) <= length(which(modulo$p535==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5371"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5371)) <= sum(!is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p538a1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p538a1)) <= sum(!is.na(modulo$p5371))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p538a2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p538a2)) >= sum(!is.na(modulo$p538a1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p539"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p539)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5401a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5401a)) <= sum(!is.na(modulo$p539))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5401b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5401b)) <= sum(!is.na(modulo$p5401a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5401c"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5401c)) <= length(which(modulo$p539==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5407a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5407a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p540t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p540t)) <= sum(!is.na(modulo$p5407a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p541a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p541a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p541b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p541b)) <= sum(!is.na(modulo$p541a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p542"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p542)) <= sum(!is.na(modulo$p541a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p543"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p543)) <= sum(!is.na(modulo$p542))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5441a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5441a)) <= sum(!is.na(modulo$p505))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5441b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5441b)) <= sum(!is.na(modulo$p5441a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p544t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p544t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p545"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p545)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p546"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p546)) <= length(which(modulo$p545==2))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p547"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p547)) <= sum(!is.na(modulo$p546))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p548"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p548)) <= length(which(modulo$p547==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p549"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p549)) <= length(which(modulo$p548==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p550"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p550)) <= sum(!is.na(modulo$p545))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p551"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p551)) <= sum(!is.na(modulo$p545))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p552"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p552)) <= sum(!is.na(modulo$p545))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p554"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p554)) <= length(which(modulo$p552==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p554r4"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p554r4)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p555"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p555)) <= length(which(modulo$p552==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5561a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5561a)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5561b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5561b)) <= length(which(modulo$p5561a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5561c"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5561c)) <= length(which(modulo$p5561a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5561d"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5561d)) <= length(which(modulo$p5561a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5561e"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5561e)) <= length(which(modulo$p5561a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p556t1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p556t1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p556t2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p556t2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5571a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5571a)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5571b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5571b)) <= length(which(modulo$p5571a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5571c"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5571c)) <= length(which(modulo$p5571a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p557t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p557t)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5581a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5581a)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5581b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5581b)) <= length(which(modulo$p5581a==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p558t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p558t)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p558a1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p558a1)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p558b1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p558b1)) <= (length(which(modulo$p558a1==1)) + 
                                              length(which(modulo$p558a2==2)) + 
                                              length(which(modulo$p558a3==3)) + 
                                              length(which(modulo$p558a4==4)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p558b2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p558b2)) <= (length(which(modulo$p558a1==1)) + 
                                              length(which(modulo$p558a2==2)) + 
                                              length(which(modulo$p558a3==3)) + 
                                              length(which(modulo$p558a4==4)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p558b3"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p558b3)) <= (length(which(modulo$p558a1==1)) + 
                                              length(which(modulo$p558a2==2)) + 
                                              length(which(modulo$p558a3==3)) + 
                                              length(which(modulo$p558a4==4)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559n_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559n_01)) == sum(!is.na(modulo$p559_01))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559_01)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559_05"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559_05)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559_06"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559_06)) <= sum(!is.na(modulo$p559_05))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559_07"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559_07)) <= sum(!is.na(modulo$p559_05))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559a_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559a_01)) <= length(which(modulo$p559_01==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559a_02"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559a_02)) <= length(which(modulo$p559_02==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559b_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559b_01)) <= length(which(modulo$p559_01==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559c_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559c_01)) <= length(which(modulo$p559_01==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559d_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559d_01)) <= length(which(modulo$p559c_01==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p559e_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p559e_01)) <= length(which(modulo$p559c_01==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p59f1_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p59f1_01)) <= length(which(modulo$p559e_01==2))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p59f2_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p59f2_01)) <= length(which(modulo$p559e_01==2))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560tr"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560tr)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560a1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560a1)) <= length(which(modulo$p560tr==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560b1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560b1)) <= length(which(modulo$p560tr==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560c1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560c1)) <= length(which(modulo$p560tr==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560d1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560d1)) <= length(which(modulo$p560c1==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560e1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560e1)) <= length(which(modulo$p560c1==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560ft1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560ft1)) <= length(which(modulo$p560e1==2))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p560ft2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p560ft2)) <= length(which(modulo$p560e1==2))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p599"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p599)) <= sum(!is.na(modulo$p501))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat05 <- nfeat
        tmpFeat05 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo22" %in% nmodulo){
        if(i == 2){
            tmpModulo22 <- read.csv(tmpFile[5])    # módulo base
            tmpNFeat22 <- dim(tmpModulo22)[2]    # número de características base
            tmpFeat22 <- colnames(tmpModulo22)    # características del módulo base
            rm(tmpModulo22)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo22")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo22")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat22    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat22, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat22)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p20001a"
        if(colnm %in% same){
            if(sum(is.na(modulo$p20001a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p20001b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p20001b)) == sum(is.na(modulo$p20001a))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p20002"
        if(colnm %in% same){
            if(sum(is.na(modulo$p20002)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p20002b1"
        if(colnm %in% same){
            if(sum(is.na(modulo$p20002b1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p20002b2"
        if(colnm %in% same){
            if(sum(is.na(modulo$p20002b2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p20002b3"
        if(colnm %in% same){
            if(sum(is.na(modulo$p20002b3)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat22 <- nfeat
        tmpFeat22 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo22_1" %in% nmodulo){
        if(i == 2){
            tmpModulo221 <- read.csv(tmpFile[6])    # módulo base
            tmpNFeat221 <- dim(tmpModulo221)[2]    # número de características base
            tmpFeat221 <- colnames(tmpModulo221)    # características del módulo base
            rm(tmpModulo221)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo22_1")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo22_1")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat221    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat221, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat221)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p2005b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2005b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2005c1"
        if(colnm %in% same){
            ex <- (c("p2005c2", "p2005c3", "p2005c4", 
                     "p2005c5", "p2005c6", "p2005c7") %in% same)
            if(!"FALSE" %in% ex){
                cat("\n--", colnm, "Ok ([p2005c(1:7)] deberían reemplazarse
                    por una sola columna)")
            }else{cat("\n--", colnm, "No cumple ([p2005c(1:7)] no existe
                      algunas variables)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2005e"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2005e)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat221 <- nfeat
        tmpFeat221 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
        }
    
    if("Modulo22_2" %in% nmodulo){
        if(i == 2){
            tmpModulo222 <- read.csv(tmpFile[7])    # módulo base
            tmpNFeat222 <- dim(tmpModulo222)[2]    # número de características base
            tmpFeat222 <- colnames(tmpModulo222)    # características del módulo base
            rm(tmpModulo222)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo22_2")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo22_2")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat222    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat222, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat222)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p2100a"
        if(colnm %in% same){
            cat("\n--", colnm, "Ok (eliminar esta variable)")
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2100b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2100b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p21001a"
        if(colnm %in% same){
            if(3*sum(is.na(modulo$p21001a)) == (sum(is.na(modulo$p21002e)) + 
                                                sum(is.na(modulo$p21002l)) + 
                                                sum(is.na(modulo$p21002n)))){
                cat("\n--", colnm, "Ok (18 variables tiene misma cantidad de NAs)")
            }else{cat("\n--", colnm, "No cumple (no tienen misma cantidad de NAs)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p21002t"
        if(colnm %in% same){
            if(sum(is.na(modulo$p21002t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat222 <- nfeat
        tmpFeat222 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo23" %in% nmodulo){
        if(i == 2){
            tmpModulo23 <- read.csv(tmpFile[8])    # módulo base
            tmpNFeat23 <- dim(tmpModulo23)[2]    # número de características base
            tmpFeat23 <- colnames(tmpModulo23)    # características del módulo base
            rm(tmpModulo23)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo23")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo23")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat23    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat23, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat23)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p2200a"
        if(colnm %in% same){
            cat("\n--", colnm, "Ok (eliminar esta variable)")
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2200b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2200b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p22001a"
        if(colnm %in% same){
            if(3*sum(is.na(modulo$p22001a)) == (sum(is.na(modulo$p22001c)) + 
                                                sum(is.na(modulo$p22002e)) + 
                                                sum(is.na(modulo$p22002h)))){
                cat("\n--", colnm, "Ok (12 variables tiene misma cantidad de NAs)")
            }else{cat("\n--", colnm, "No cumple (no tienen misma cantidad de NAs)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p22002t"
        if(colnm %in% same){
            if(sum(is.na(modulo$p22002t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat23 <- nfeat
        tmpFeat23 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo24" %in% nmodulo){
        if(i == 2){
            tmpModulo24 <- read.csv(tmpFile[9])    # módulo base
            tmpNFeat24 <- dim(tmpModulo24)[2]    # número de características base
            tmpFeat24 <- colnames(tmpModulo24)    # características del módulo base
            rm(tmpModulo24)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo24")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo24")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat24    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat24, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat24)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p2300a"
        if(colnm %in% same){
            cat("\n--", colnm, "Ok (eliminar esta variable)")
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2300b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2300b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p23001"
        if(colnm %in% same){
            if(3*sum(is.na(modulo$p23001)) == (sum(is.na(modulo$p23002a)) + 
                                               sum(is.na(modulo$p23002b)) + 
                                               sum(is.na(modulo$p23002c)))){
                cat("\n--", colnm, "Ok (4 variables tiene misma cantidad de NAs)")
            }else{cat("\n--", colnm, "No cumple (no tienen misma cantidad de NAs)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p23002t"
        if(colnm %in% same){
            if(sum(is.na(modulo$p23002t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat24 <- nfeat
        tmpFeat24 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo25" %in% nmodulo){
        if(i == 2){
            tmpModulo25 <- read.csv(tmpFile[10])    # módulo base
            tmpNFeat25 <- dim(tmpModulo25)[2]    # número de características base
            tmpFeat25 <- colnames(tmpModulo25)    # características del módulo base
            rm(tmpModulo25)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo25")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo25")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat25    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat25, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat25)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat25 <- nfeat
        tmpFeat25 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo26" %in% nmodulo){
        if(i == 2){
            tmpModulo26 <- read.csv(tmpFile[11])    # módulo base
            tmpNFeat26 <- dim(tmpModulo26)[2]    # número de características base
            tmpFeat26 <- colnames(tmpModulo26)    # características del módulo base
            rm(tmpModulo26)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo26")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo26")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat26    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat26, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat26)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p2500a"
        if(colnm %in% same){
            cat("\n--", colnm, "Ok (eliminar esta variable)")
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2500b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2500b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p25001a"
        if(colnm %in% same){
            if(3*sum(is.na(modulo$p25001a)) <= (sum(is.na(modulo$p25001b)) + 
                                                sum(is.na(modulo$p25002a1)) + 
                                                sum(is.na(modulo$p25002o3)))){
                cat("\n--", colnm, "Ok (31 variables tiene misma cantidad de NAs +- 1)")
            }else{cat("\n--", colnm, "No cumple (no tienen misma cantidad de NAs)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p25002t"
        if(colnm %in% same){
            if(sum(is.na(modulo$p25002t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat26 <- nfeat
        tmpFeat26 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo27" %in% nmodulo){
        if(i == 2){
            tmpModulo27 <- read.csv(tmpFile[12])    # módulo base
            tmpNFeat27 <- dim(tmpModulo27)[2]    # número de características base
            tmpFeat27 <- colnames(tmpModulo27)    # características del módulo base
            rm(tmpModulo27)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo27")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo27")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat27    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat27, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat27)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p2600a"
        if(colnm %in% same){
            cat("\n--", colnm, "Ok (eliminar esta variable)")
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2600b"
        if(colnm %in% same){
            if(sum(is.na(modulo$p2600b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p26001a"
        if(colnm %in% same){
            if(3*sum(is.na(modulo$p26001a)) == (sum(is.na(modulo$p26001c)) + 
                                                sum(is.na(modulo$p26002a)) + 
                                                sum(is.na(modulo$p26002h)))){
                cat("\n--", colnm, "Ok (10 variables tiene misma cantidad de NAs)")
            }else{cat("\n--", colnm, "No cumple (no tienen misma cantidad de NAs)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p26002t"
        if(colnm %in% same){
            if(sum(is.na(modulo$p26002t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat27 <- nfeat
        tmpFeat27 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo28" %in% nmodulo){
        if(i == 2){
            tmpModulo28 <- read.csv(tmpFile[13])    # módulo base
            tmpNFeat28 <- dim(tmpModulo28)[2]    # número de características base
            tmpFeat28 <- colnames(tmpModulo28)    # características del módulo base
            rm(tmpModulo28)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo28")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo28")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat28    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat28, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat28)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat28 <- nfeat
        tmpFeat28 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo37" %in% nmodulo){
        if(i == 2){
            tmpModulo37 <- read.csv(tmpFile[14])    # módulo base
            tmpNFeat37 <- dim(tmpModulo37)[2]    # número de características base
            tmpFeat37 <- colnames(tmpModulo37)    # características del módulo base
            rm(tmpModulo37)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo37")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo37")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat37    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat37, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat37)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p706a1"
        if(colnm %in% same){
            if(sum(is.na(modulo$p706a1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p706a2"
        if(colnm %in% same){
            if(2*sum(is.na(modulo$p706a2)) == (sum(is.na(modulo$p706a3)) + 
                                               sum(is.na(modulo$p706a4)))){
                cat("\n--", colnm, "Ok (3 variables tiene misma cantidad de NAs)")
            }else{cat("\n--", colnm, "No cumple (no tienen misma cantidad de NAs)")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat37 <- nfeat
        tmpFeat37 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo77" %in% nmodulo){
        if(i == 2){
            tmpModulo77 <- read.csv(tmpFile[15])    # módulo base
            tmpNFeat77 <- dim(tmpModulo77)[2]    # número de características base
            tmpFeat77 <- colnames(tmpModulo77)    # características del módulo base
            rm(tmpModulo77)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo77")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo77")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat77    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat77, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat77)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "e1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e1b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e1b)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e3"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e3)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e4a1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e4a1)) >= sum(!is.na(modulo$e3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e4b1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e4b1)) <= length(which(modulo$e4a1==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e5"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e5)) <= sum(!is.na(modulo$e1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e6a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e6a)) <= sum(!is.na(modulo$e1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e8a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e8a)) <= sum(!is.na(modulo$e1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e14t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e14t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e15"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e15)) <= sum(!is.na(modulo$e14t))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e15t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e15t)) <= sum(!is.na(modulo$e15))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e16t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e16t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e17t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e17t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e18"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e18)) <= sum(!is.na(modulo$e17t))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e20t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e20t)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e21"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e21)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e23st"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e23st)) <= sum(!is.na(modulo$e1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e24t"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e24t)) >= sum(!is.na(modulo$e1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(!is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat77 <- nfeat
        tmpFeat77 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo77_1" %in% nmodulo){
        if(i == 2){
            tmpModulo771 <- read.csv(tmpFile[16])    # módulo base
            tmpNFeat771 <- dim(tmpModulo771)[2]    # número de características base
            tmpFeat771 <- colnames(tmpModulo771)    # características del módulo base
            rm(tmpModulo771)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo77_1")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo77_1")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat771    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat771, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat771)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat771 <- nfeat
        tmpFeat771 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo77_2" %in% nmodulo){
        if(i == 2){
            tmpModulo772 <- read.csv(tmpFile[17])    # módulo base
            tmpNFeat772 <- dim(tmpModulo772)[2]    # número de características base
            tmpFeat772 <- colnames(tmpModulo772)    # características del módulo base
            rm(tmpModulo772)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo77_2")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo77_2")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat772    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat772, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat772)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "e23a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$e23a)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat772 <- nfeat
        tmpFeat772 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo77_3" %in% nmodulo){
        if(i == 2){
            tmpModulo773 <- read.csv(tmpFile[18])    # módulo base
            tmpNFeat773 <- dim(tmpModulo773)[2]    # número de características base
            tmpFeat773 <- colnames(tmpModulo773)    # características del módulo base
            rm(tmpModulo773)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo77_3")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo77_3")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat773    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat773, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat773)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "e24c"
        if(colnm %in% same){
            if(sum(is.na(modulo$e24c)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e24d"
        if(colnm %in% same){
            if(sum(is.na(modulo$e24d)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e24e1"
        if(colnm %in% same){
            if(sum(is.na(modulo$e24e1)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e24e2"
        if(colnm %in% same){
            if(sum(is.na(modulo$e24e2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e24f"
        if(colnm %in% same){
            if(sum(is.na(modulo$e24f)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "e24g"
        if(colnm %in% same){
            if(sum(is.na(modulo$e24g)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat773 <- nfeat
        tmpFeat773 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo77_4" %in% nmodulo){
        if(i == 2){
            tmpModulo774 <- read.csv(tmpFile[19])    # módulo base
            tmpNFeat774 <- dim(tmpModulo774)[2]    # número de características base
            tmpFeat774 <- colnames(tmpModulo774)    # características del módulo base
            rm(tmpModulo774)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo77_4")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo77_4")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat774    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat774, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat774)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "factora0"
        if(colnm %in% same){
            if(sum(is.na(modulo$factora0)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat774 <- nfeat
        tmpFeat774 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo84_1" %in% nmodulo){
        if(i == 2){
            tmpModulo841 <- read.csv(tmpFile[20])    # módulo base
            tmpNFeat841 <- dim(tmpModulo841)[2]    # número de características base
            tmpFeat841 <- colnames(tmpModulo841)    # características del módulo base
            rm(tmpModulo841)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo84_1")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo84_1")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat841    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat841, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat841)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p803"
        if(colnm %in% same){
            if(sum(is.na(modulo$p803)) == 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p804"
        if(colnm %in% same){
            if(sum(is.na(modulo$p804)) == 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p805"
        if(colnm %in% same){
            if(sum(is.na(modulo$p805)) == 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "facpob07"
        if(colnm %in% same){
            if(sum(is.na(modulo$facpob07)) == 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat841 <- nfeat
        tmpFeat841 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo85" %in% nmodulo){
        if(i == 2){
            tmpModulo85 <- read.csv(tmpFile[21])    # módulo base
            tmpNFeat85 <- dim(tmpModulo85)[2]    # número de características base
            tmpFeat85 <- colnames(tmpModulo85)    # características del módulo base
            rm(tmpModulo85)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo85")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo85")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat85    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat85, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat85)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p1_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p1_01)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p1_02"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p1_02)) <= sum(!is.na(modulo$p1_01))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2_1_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p2_1_01)) <= sum(!is.na(modulo$p1_01))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p2_2_01"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p2_2_01)) <= length(which(modulo$p2_1_01==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p3"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p3)) <= sum(!is.na(modulo$p1_01))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p4"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p4)) <= length(which(modulo$p3==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p5_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p5_1)) <= length(which(modulo$p3==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p6"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p6)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p7"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p7)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p8_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p8_1)) <= (length(which(modulo$p7==1)) + 
                                            length(which(modulo$p7==2)))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p9"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p9)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p10_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p10_1)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p10_2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p10_2)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p11_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p11_1)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p12"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p12)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p14"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p14)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p15"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p15)) <= sum(!is.na(modulo$p14))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p16"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p16)) <= length(which(modulo$p15==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p17"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p17)) <= length(which(modulo$p15==2))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p18"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p18)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p19"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p19)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p20_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p20_1)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p21"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p21)) <= sum(!is.na(modulo$p3))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p22"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p22)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p23"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p23)) <= sum(!is.na(modulo$p22))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p23a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p23a)) <= length(which(modulo$p23==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p23b"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p23b)) <= length(which(modulo$p23==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "factor07"
        if(colnm %in% same){
            if(sum(!is.na(modulo$factor07)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat85 <- nfeat
        tmpFeat85 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    
    if("Modulo85_1" %in% nmodulo){
        if(i == 2){
            tmpModulo851 <- read.csv(tmpFile[22])    # módulo base
            tmpNFeat851 <- dim(tmpModulo851)[2]    # número de características base
            tmpFeat851 <- colnames(tmpModulo851)    # características del módulo base
            rm(tmpModulo851)
        }
        
        modulo <- read.csv(indFiles[which(nmodulo == "Modulo85_1")], header = T)
        cat("\n\n\nVariaciones encontradas en la estructura para el: ", 
            nmodulo[which(nmodulo == "Modulo85_1")])
        cat("(", dim(modulo)[1], ",", dim(modulo)[2], ")")
        
        #### Primero revisamos si tienen las mismas características con el año anterior
        nfeat <- dim(modulo)[2]    # número de características
        varnfeat <- nfeat - tmpNFeat851    # diferencia en el número de característica
        cat("\n\n**La diferencia de características respecto al año anterior es:", varnfeat)
        
        feat <- colnames(modulo)    # características del módulo
        same <- Reduce(intersect, list(tmpFeat851, feat))    # características que se mantienen
        cat("\n**La características que se mantienen son las siguientes: \n[", same, "]")
        varfeat <- setdiff(feat, tmpFeat851)  # características que no se encuentran en el modulo actual
        cat("\n**Las características nuevas son: \n[", varfeat, "]")
        
        #### Análisis de las características que se mantienen de acuerdo al módulo 2007
        colnm <- "p32"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p32)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p33_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p33_1)) >= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p33_2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p33_2)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p34"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p34)) <= length(which(modulo$p33_2==1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p37"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p37)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p38"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p38)) <= sum(!is.na(modulo$p37))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p38a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p38a)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p39a"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p39a)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p40_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p40_1)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p41"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p41)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p42_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p42_1)) <= sum(!is.na(modulo$p41))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p43"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p43)) <= sum(!is.na(modulo$p42_1))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p45_1"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p45_1)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p45_2"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p45_2)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p46"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p46)) <= sum(!is.na(modulo$p32))){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        colnm <- "p47"
        if(colnm %in% same){
            if(sum(!is.na(modulo$p47)) >= 0){
                numCumple <- numCumple + 1
                cat("\n--", colnm, "Ok")
            }else{cat("\n--", colnm, "No cumple")}
        }else{cat("\n--", colnm, "No existe en este módulo!")}
        
        tmpNFeat851 <- nfeat
        tmpFeat851 <- feat
        
        cat("\n**Cantidad de variables que cumplen con las reglas: ", numCumple, "\n\n")
        numCumple <- c(0)
    }
    readline(prompt = "Press enter to see the next year")
}


## Check how many missing values there are
#abvNames <- c()
nas <- c()
tnas <- data.frame()
for(i in 1:length(modulo)){
    #abvNames <- rbind(abvNames, substr(colnames(modulo)[i], 1, 4))
    nas <- sum(is.na(modulo[,i]))
    if(nas != 0){
        tnas <- rbind(tnas, cbind(colnames(modulo)[i], nas))
    }
}
colnames(tnas) <- c("feature", "NA's")
print(tnas)

## This verify the number of different rows
subs <- modulo[,1:6]
ia <- subs[1,]
nind <- c(1)
for(i in 2:dim(subs)[1]){
  eq <- ia==subs[i,]
  if('FALSE' %in% eq == TRUE){
    nind <- nind + 1
  }
  ia <- subs[i,]
}
print(nind)




