from XGB_cases import XGB_cases

files_delta = ['case1', 'case2']

OUT = 10
IN = 5

for file in files_delta:
    print('_' + file + '_running...')
    XGB_cases(file, OUT, IN)
