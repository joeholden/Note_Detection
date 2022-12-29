import matplotlib.pyplot as plt
from pydub import AudioSegment
import struct
import scipy
from scipy import fftpack, signal
import numpy as np
import pandas as pd
from lookup_note import return_note_from_freq, return_frequency_of_note

path = 'C:/Users/joema/Documents/Sound recordings/c chord.m4a'


def find_frequencies(audio_path, time_limits=None, prom=7, show_plot=True, plot_width=(25, 4200),
                     plot_harmonic_vline=False, num_harmonics=6):
    """
    Takes in Audio file and outputs a list containing information about frequency peaks in FFT.

    :param audio_path: path to the audio file
    :param time_limits:
    :param prom: Prominence (Intensity) cut off for peak detection
    :param show_plot: Boolean True, False
    :param plot_width: The frequency range in Hertz that you want displayed. Taken as a tuple
    :param plot_harmonic_vline: Do you want vertical lines drawn on harmonics of main peak
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

    # Find peaks
    peaks = scipy.signal.find_peaks(intensity_y, prominence=prom)  # First member of object holds indices of peaks
    all_peak_information = list(zip(xf[peaks[0]], peaks[1]['prominences']))
    all_peak_information.sort(key=lambda k: k[1], reverse=True)  # inplace operation, sort by intensity of peak
    all_peak_information = [(i, j, return_note_from_freq(i)) for (i, j) in all_peak_information]  # add note guesses
    print(all_peak_information)

    # Plot Scatter Points of Peaks
    frequencies_of_peaks = []
    for peak, value in enumerate(peaks[0]):
        plt.scatter(xf[peaks[0]], [intensity_y[i] for i in peaks[0]], s=50, color='#ED7014')
        frequencies_of_peaks.append(round(xf[peaks[0]][peak], 2))

    prominences = peaks[1]['prominences']
    index_highest_peak = peaks[0][list(prominences).index(max(prominences))]
    m = xf[index_highest_peak]

    # Draw Vertical Lines at Each Harmonic Peak
    style_vline = {"vline_color": "#7953A9", "ls": "--", "dashes": (10, 10)}
    if plot_harmonic_vline:
        for i in range(1, num_harmonics + 1):
            plt.axvline(m * i,
                        color=style_vline['vline_color'],
                        linestyle=style_vline['ls'],
                        dashes=style_vline['dashes'])

    # Style Plot
    plt.xlim(plot_width[0], plot_width[1])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.set_tick_params(size=5, width=2)
    ax.yaxis.set_tick_params(size=5, width=2)
    plt.title("FFT Audio", fontsize=22)
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)

    if show_plot:
        plt.show()

    return all_peak_information


find_frequencies(path, plot_width=(0, 2000))
