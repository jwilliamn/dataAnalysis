import os


def sum(a, b):
	path = os.path.abspath(__file__)
	return [path,a + b]