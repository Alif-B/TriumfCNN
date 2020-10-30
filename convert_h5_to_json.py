
import json

# file containing the list of datafiles
# that needs to be converted from h5 into json
filename = 'h5_dataset.txt'

event_arguments = {}

with open(filename) as fh:
    for line in fh:
        filename = line.strip('\n')

        argument = line.strip('.txt\n')
        event_arguments[argument] = filename


# dictionary where the data of each type
# will be stored
event_h5_data = {}

# iterating through all types of data
# and writing them into a single json file
for key, values in event_arguments.items():
    with open(values, 'r') as file:
        # reads each line and trims of extra the spaces
        contents = file.readlines()
        event_data = []

        for data in contents:
            # strips new lines from the data
            event_data.append(data.strip('\n'))

        event_h5_data[key] = event_data

# creating json file
out_file = open("h5_to_json_converted.json", "w")
json.dump(event_h5_data, out_file, indent=4, sort_keys=False)
out_file.close()

