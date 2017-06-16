__author__ = 'williamn'
# Python script to download financial information from NYSE - financials.morningstar.com webpage

import os
from urllib.error import URLError
from urllib.request import urlopen


url = ["http://tramite.midis.gob.pe/institucional/aplicativos/oad/sitradocV2/archivos/450254.pdf",
"http://tramite.midis.gob.pe/institucional/aplicativos/oad/sitradocV2/archivos/450257.pdf",
"http://tramite.midis.gob.pe/institucional/aplicativos/oad/sitradocV2/archivos/450259.pdf",
"http://tramite.midis.gob.pe/institucional/aplicativos/oad/sitradocV2/archivos/450259.pdf"
]

def download_file(download_url):
    pdfname = os.path.basename(download_url)
    
    try:
        response = urlopen(download_url)
        file = open(pdfname, 'wb')
        file.write(response.read())
        file.close()
        print("pdfname: ", pdfname, "Completed")
    except URLError:
        print("Bad url")

def main():
    #download_file("http://tramite.midis.gob.pe/institucional/aplicativos/oad/sitradocV2/archivos/450254.pdf")
    for i in range(0,len(url)):
        #print("url: ", url[i])
        download_file(url[i])

if __name__ == "__main__":
    main()
