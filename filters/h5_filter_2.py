from h5_dataset import H5Dataset
from itertools import groupby

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
    pass


if __name__ == "__main__":
    for event in h5:
        print(event)


