import pandas as pd
import unicodedata
import numpy as np

frequency_table = pd.read_excel("key frequencies.xlsx")

# Get a Matrix of Harmonics
harmonics = np.array(frequency_table['Frequency (Hz)'])
harmonic_matrix = harmonics.copy()
for i in range(2, 7):
    new_h = harmonics * i
    harmonic_matrix = np.column_stack((harmonic_matrix, new_h))

harmonics = pd.DataFrame(harmonic_matrix, columns=['1', '2', '3', '4', '5', '6'])
harmonics.index += 1


# Prints the first key's harmonics.
print(harmonics[0:1])


def find_all_notes(frequencies_from_file, frequency_matrix, tolerance=1.05):
    def is_valid(input_1, input_2):
        if -1 * tolerance * input_1 <= input_2 <= tolerance * input_1:
            return True
        else:
            return False
    for row in frequency_matrix:
        r = frequency_matrix[row:row+1]





def return_frequency_of_note(note):
    notes = list(frequency_table['Scientific Name'])
    frq = list(frequency_table['Frequency (Hz)'])

    index = notes.index(note)
    return frq[index]


def return_note_from_freq(frequency, tolerance=0.03):
    notes = list(frequency_table['Scientific Name'])
    frq = list(frequency_table['Frequency (Hz)'])

    possible_notes = []
    for index, f in enumerate(frq):
        if (abs(frequency - f) / f) <= tolerance:
            n = notes[index]
            n = n.replace(u'\xa0', u' ')
            possible_notes.append(n)
    return possible_notes


return_note_from_freq(261.6)