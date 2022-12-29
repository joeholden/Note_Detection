import matplotlib.pyplot as plt
from pydub import AudioSegment
import struct
import scipy
from scipy import fftpack, signal
import numpy as np
import pandas as pd

path = 'C:/Users/joema/Documents/Sound recordings/Recording (10).m4a'
frequency_table = pd.read_excel("key frequencies.xlsx")

# Starting at the highest intensity notes, check for 3 overtones. If present to a degree of 2%, you found a note
# being played. Play many notes on the piano as pure tones and see the ratio in intensities for overtones. Use this
# condition as well


def find_frequencies(audio_path, time_limits=None, show_plot=True, plot_width=(0, 4200), num_harmonics=6):
    """
    Takes in Audio file and outputs a list containing information about frequency peaks in FFT.

    :param audio_path: path to the audio file
    :param time_limits:
    :param show_plot: Boolean True, False
    :param plot_width: The frequency range in Hertz that you want displayed. Taken as a tuple
    :param num_harmonics: The number of harmonics that you want displayed as vertical lines on the plot.
           Includes fundamental frequency
    :return: Output is a list containing tuples where the first element is the frequency of a peak and the second is
             the intensity of that peak. Peaks are ordered descending order by intensity.
    """
    # Get Information About Audio File
    audio = AudioSegment.from_file(audio_path)
    raw = audio.raw_data
    sample_rate = audio.frame_rate
    sample_size = audio.sample_width
    channels = audio.channels

    # Format Audio and Run Fourier Transform
    fmt = "%ih" % audio.frame_count() * channels
    amplitudes = struct.unpack(fmt, raw)

    S = pd.Series(amplitudes)
    N = len(S)  # Number of Samples
    t = np.random.uniform(0.0, 1.0, N)  # Assuming the time start is 0.0 and time end is 1.0
    T = 1 / sample_rate  # sample spacing
    x = np.linspace(0.0, N * T, N)

    # I believe the raw file repeats the frequencies reflected across y axis
    yf = scipy.fftpack.fft(S)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    intensity_y = 2.0 / N * np.abs(yf[:N // 2])
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 6))

    plt.plot(xf, intensity_y, color="#663399")
    plt.xlim(plot_width[0], plot_width[1])

    # Find peaks
    peaks = scipy.signal.find_peaks(intensity_y, prominence=40)  # First member of object holds indices of peaks
    all_peak_information = list(zip(peaks[0], peaks[1]['prominences']))

    frequencies_of_peaks = []
    for peak, value in enumerate(peaks[0]):
        plt.scatter(xf[peaks[0]], [intensity_y[i] for i in peaks[0]], s=50)
        frequencies_of_peaks.append(round(xf[peaks[0]][peak], 2))

    prominences = peaks[1]['prominences']
    index_highest_peak = peaks[0][list(prominences).index(max(prominences))]
    m = xf[index_highest_peak]

    # Draw Vertical Lines at Each Harmonic Peak
    vline_color = "#7953A9"
    ls = "--"
    dashes_tuple = (10, 10)

    def draw_vlines(num_harmonics):
        for i in range(1, num_harmonics+1):
            plt.axvline(m*i, color=vline_color, linestyle=ls, dashes=dashes_tuple)

    draw_vlines(num_harmonics)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_tick_params(size=5, width=2)
    ax.yaxis.set_tick_params(size=5, width=2)
    plt.title("FFT Audio", fontsize=22)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)

    plt.show()

    all_peak_information.sort(key=lambda k: k[1], reverse=True)  # inplace operation
    print(all_peak_information)
    return all_peak_information


find_frequencies(path, plot_width=(0, 4200))
