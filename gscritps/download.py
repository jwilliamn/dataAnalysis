__author__ = 'williamn'
# Python script to download financial information from NYSE - financials.morningstar.com webpage

import urllib2
import csv
import StringIO
import os

url = "http://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t={}" \
      "&region=usa&culture=en-US&cur=USD&reportType={}&period={}&dataType=R" \
      "&order=asc&columnYear=5&rounding=3&view=raw&denominatorView=raw&number=2"

stock = ['CRPJF', 'TKECY', 'KAEPY', 'SSEZF', 'CPYYF', 'IBDSF', 'TERRF', 'ELCPF', 'EDPFY', 'ESOCF',
 'CPYYY','ELPSF','STRNY','IBDRY','SVTRF','MAEOY','TEZNY','GASNF','ENLAY','SZEVY',
 'NGGTF','ELEZF','ABZPF','ECIFF','RWNEF','SZEVF','SSEZY','AGLNF','HOKCF','ED',
 'RDEIF','NEE','NGG','HOKCY','HPIFF','DCUC','UUGWF','SCG','PPL','SO','CLPHY',
 'PCG','RDEIY','GGDVF','ECIFY','CLPHF','AES','AEP','XEL','GDSZF','PEG','ES','GGDVY',
 'GASNY','CMS','LNT','EIX','HGKGF','DTE','GDFZY','D','KEP','WEC','AGLNY','AWK',
 'PPAAF','AEE','HGKGY','EXC','SRE','CLPXY','EXCU','CRPJY','DUK','UUGRY','FE','ENAKF',
 'EOC','CGHOF','ETR','HNP','CNP','FRTSF','APAJF','EONGY','TBLEY','ENI','BEP','TKGSY',
 'FOJCF','NRG','FOJCY','RWEOY','GAILF','CDUAF','TKGSF','PPAAY','CLPXF','RWNFF','TNABY']

report_type = ['is', 'bs', 'cf']
period = 12  # In months, 12 = annual, 3 = quarterly.

path = '/home/williamn/Repository/data/financialData/foreignCompanies/sector/utilities'

for stocki in stock:
    for reptypej in report_type:
        request = urllib2.urlopen(url.format(stocki, reptypej, period))

        fileInfo = request.info()
        data = request.read()

        f = StringIO.StringIO(data)
        reader = csv.reader(f)

        filename = stocki + '_' + reptypej + '.csv'
        if os.path.exists(path) == 0:
            os.makedirs(path)
        newFile = open(os.path.join(path, filename), 'w')
        fileWriter = csv.writer(newFile)

        for row in reader:
            if len(row) != 1:
                fileWriter.writerow(row)
                # print row
        newFile.close()

    print stocki + ' done!'
