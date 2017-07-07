# Printing numbers 50 - 200 without loops

def printNum(ini):
	if ini >= 50:
		printNum(ini - 1)
		print(ini)
	
printNum(200)

# Desired output: [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
matriz = [[x*y for y in range(1,4)] for x in range(1,4)]