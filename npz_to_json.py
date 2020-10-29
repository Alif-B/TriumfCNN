import numpy as np
import json

data = np.load("event998.npz", allow_pickle=True)
file = open('npz.JSON', 'w')

i = 0
while i < 3000:
    file.write("{ \n")
    for x in data:
        file.write(f"    '{x}': {data[x][i]}\n")
    file.write("}\n")
    i += 1
file.close()



# row = data.files
# np.set_printoptions(threshold=np.inf)
# print(data['root_file'])
# sys.stdout=open("test.txt","w")
# for i in row:
#     print("--------------------------")
#     print(data[i])
# sys.stdout.close()

# import numpy as np
# import h5py
# filename = 'multimuonfile.h5'
# f = h5py.File(filename, 'r')
# np.savetxt('h5file.txt', f['dataset'][...])
# f.close()