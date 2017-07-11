import os
from test2 import sum

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.dirname(os.path.abspath(__file__))

print('base_dir', BASE_DIR)
print('path to parent dir', path)
print('sum', sum(2,3))