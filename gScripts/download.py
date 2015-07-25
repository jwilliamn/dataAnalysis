# Python script to download financial information from NYSE - finance morningstar webpage
import urllib2
import csv
import StringIO

url = "http://financials.morningstar.com/ajax/ReportProcess4CSV.html?&t={}" \
      "&region=usa&culture=en-US&cur=USD&reportType={}&period={}&dataType=R" \
      "&order=asc&columnYear=5&rounding=3&view=raw&denominatorView=raw&number=2"

stock = "DUK"
rtype = "bs"
period = 12 #In months, 12 = annual, 3 = quarterly, etc.

request = urllib2.urlopen(url.format(stock, rtype, period))

filename = request.info()
data = request.read()

f = StringIO.StringIO(data)
print f[1]
reader = csv.reader(f)
new_file = open(stock + '_' + rtype + '.csv', 'w')
f_writer = csv.writer(new_file)

print new_file

for row in reader:
    #f_writer.writerow(row)
    print row