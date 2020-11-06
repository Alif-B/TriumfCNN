import numpy as np
import json

npz_file = "NonPyFiles/event998.npz"
JSON_file = "NonPyFiles/npz.JSON"
data = np.load(npz_file, allow_pickle=True)
file = open(JSON_file, 'w')

i = 0
while i < 3000:
    file.write("{ \n")
    for x in data:
        file.write(f"    '{x}': {data[x][i]}\n")
    file.write("}\n")
    i += 1
file.close()
