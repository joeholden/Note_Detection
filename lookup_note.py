import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import random
import math

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


def find_multiples(all_peaks_info, tolerance=0.03):
    peak_harmonics = {}
    enumerated_all_peaks_info = list(enumerate(all_peaks_info))

    passes = 1
    while len(enumerated_all_peaks_info) > 0:
        random_color_float = random.uniform(0, 1)
        cmap = cm.get_cmap('hsv')
        color = cmap(random_color_float)
        color = color[0:3]

        target = enumerated_all_peaks_info[0][1][0]  # get target peak to find multiples of

        # For every item remaining in all_peaks_copy, find error from being an integer multiple of target
        # First term retains tag from enumerated_all_peaks_info index
        divided = [(i[0], i[1][0]/target) for i in enumerated_all_peaks_info]
        error = [(i[0], (i[1]-round(i[1], 0))/round(i[1], 0)) if i[1] > 0.5 else None for i in divided]

        plot_points = []
        for e in error:
            try:
                if abs(e[1]) < tolerance:
                    original_position = e[0]
                    for peak in enumerated_all_peaks_info:
                        if peak[0] == original_position:
                            plot_points.append((peak[1][0], peak[1][1]))
                            enumerated_all_peaks_info.remove(peak)
            except TypeError:
                pass  # This was a None Entry from error
        plt.scatter([j[0] for j in plot_points], [j[1] for j in plot_points],
                    color=color, label=return_note_from_freq(target)[0] + f"\nNumber of Harmonics:{len(plot_points)}")
        passes += 1
    print(enumerated_all_peaks_info)
    plt.legend()
    plt.show()
