from h5_dataset import H5Dataset

filepath = "../NonPyFiles/event998.h5"
h5 = H5Dataset(filepath)


def get_the_criteria():
    """ Gets the criteria on which user wants events """

    pref = {}

    pref_label = input("Preferred Label Number: ")
    pref_energy = input("Preferred Energy: ")
    pref_angle_1 = input("Preferred Angle 1: ")
    pref_angle_2 = input("Preferred Angle 2: ")
    pref_position_x = input("Preferred Position X: ")
    pref_position_y = input("Preferred Position Y: ")
    pref_position_z = input("Preferred Position Z: ")
    pref_event_id = input("Preferred Event ID: ")
    pref_root_file = input("Preferred root file: ")
    pref_indices = input("Preferred indices: ")

    preference = {
        'label': pref_label,
        'energy': pref_energy,
        'angle1': pref_angle_1,
        'angle2': pref_angle_2,
        'position_x': pref_position_x,
        'position_y': pref_position_y,
        'position_z': pref_position_z,
        'event': pref_event_id,
        'root_file': pref_root_file,
        'indices': pref_indices
    }

    for criteria in preference:
        if preference[criteria] != '':
            pref[criteria] = preference[criteria]

    return pref


def find_sample():
    """ It uses the fields of the pref dictionary to filter and returns matching samples"""
    pref = get_the_criteria()

    filtered_events = []
    temp_events = h5
    if 'label' in pref:
        [filtered_events.append(event) for event in temp_events if int(event['labels']) == int(pref['label'])]
        temp_events = filtered_events
        filtered_events = []
    if 'energy' in pref:
        [filtered_events.append(event) for event in temp_events if float(event['energies'][0]) == float(pref['energy'])]
        temp_events = filtered_events
        filtered_events = []
    if 'angle1' in pref:
        [filtered_events.append(event) for event in temp_events if float(event['angles'][0]) == float(pref['angle1'])]
        temp_events = filtered_events
        filtered_events = []
    if 'angle2' in pref:
        [filtered_events.append(event) for event in temp_events if float(event['angles'][1]) == float(pref['angle2'])]
        temp_events = filtered_events
        filtered_events = []
    if 'position_x' in pref:
        [filtered_events.append(event) for event in temp_events if float(event['positions'][0]) == float(pref['position_x'])]
        temp_events = filtered_events
        filtered_events = []
    if 'position_y' in pref:
        [filtered_events.append(event) for event in temp_events if float(event['positions'][1]) == float(pref['position_y'])]
        temp_events = filtered_events
        filtered_events = []
    if 'position_z' in pref:
        [filtered_events.append(event) for event in temp_events if float(event['positions'][2]) == float(pref['position_z'])]
        temp_events = filtered_events
        filtered_events = []
    if 'event' in pref:
        [filtered_events.append(event) for event in temp_events if int(event['event_ids']) == int(pref['event'])]
        temp_events = filtered_events
        filtered_events = []
    if 'root_file' in pref:
        [filtered_events.append(event) for event in temp_events if event['root_file'] == pref['root_file']]
        temp_events = filtered_events
        filtered_events = []
    if 'indices' in pref:
        [filtered_events.append(event) for event in temp_events if int(event['indices']) == int(pref['indices'])]
        temp_events = filtered_events

    return temp_events


if __name__ == "__main__":
    print(h5[5])

    sample1 = find_sample()
    [print(i) for i in sample1]

    sample2 = find_sample()
    [print(i) for i in sample2]
