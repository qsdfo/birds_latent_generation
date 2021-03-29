import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from avgn.signalprocessing.spectrogramming_scipy import _amplitude_to_db, _normalize, preemphasis, spectrogram_sp, _min_level_db
from tqdm import tqdm
import numpy as np
from scipy import ndimage, signal
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import gridspec


def contiguous_regions(condition):
    """
    Compute contiguous region of binary value (e.g. silence in waveform) to
        ensure noise levels are sufficiently low

    Arguments:
        condition {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    idx = []
    i = 0
    while i < len(condition):
        x1 = i + condition[i:].argmax()
        try:
            x2 = x1 + condition[x1:].argmin()
        except:
            x2 = x1 + 1
        if x1 == x2:
            if condition[x1] == True:
                x2 = len(condition)
            else:
                break
        idx.append([x1, x2])
        i = x2
    return idx


def dynamic_threshold_segmentation(
    vocalization,
    rate,
    min_level_db_floor=-40,
    db_delta=5,
    n_fft=1024,
    hop_length_ms=1,
    win_length_ms=5,
    ref_level_db=20,
    pre=0.97,
    silence_threshold=0.05,
    min_silence_for_spec=0.1,
    max_vocal_for_spec=1.0,
    min_syllable_length_s=0.05,
    spectral_range=None,
    verbose=False,
):
    """
    computes a spectrogram from a waveform by iterating through thresholds
         to ensure a consistent noise level

    Arguments:
        vocalization {[type]} -- waveform of song
        rate {[type]} -- samplerate of datas

    Keyword Arguments:
        min_level_db_floor {int} -- highest number min_level_db is allowed to reach dynamically (default: {-40})
        db_delta {int} -- delta in setting min_level_db (default: {5})
        n_fft {int} -- FFT window size (default: {1024})
        hop_length_ms {int} -- number audio of frames in ms between STFT columns (default: {1})
        win_length_ms {int} -- size of fft window (ms) (default: {5})
        ref_level_db {int} -- reference level dB of audio (default: {20})
        pre {float} -- coefficient for preemphasis filter (default: {0.97})
        min_syllable_length_s {float} -- shortest expected length of syllable (default: {0.1})
        min_silence_for_spec {float} -- shortest expected length of silence in a song (used to set dynamic threshold) (default: {0.1})
        silence_threshold {float} -- threshold for spectrogram to consider noise as silence (default: {0.05})
        max_vocal_for_spec {float} -- longest expected vocalization in seconds  (default: {1.0})
        spectral_range {[type]} -- spectral range to care about for spectrogram (default: {None})
        verbose {bool} -- display output (default: {False})


    Returns:
        [results] -- [dictionary of results]
    """

    # does the envelope meet the standards necessary to consider this a bout
    envelope_is_good = False

    # Don't worry about the name, since _mel_basis was None, this is the spectrogram
    win_length = n_fft if win_length_ms is None else int(win_length_ms / 1000 * rate)
    hop_length = win_length // 4 if hop_length_ms is None else int(hop_length_ms / 1000 * rate)
    overlap_length = win_length - hop_length
    preemphasis_y = preemphasis(x=vocalization, pre_emphasis=pre)
    _, time_bins, S = signal.stft(x=preemphasis_y, fs=rate, window='hann', nperseg=win_length, noverlap=overlap_length,
                                 nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=True, axis=-1)
    spec_orig = np.abs(S)
    ####################################################
    #################################################### T  E  S  T  LIBROSA
    # preemphasis_y = preemphasis(x=vocalization, pre_emphasis=pre)
    # hop_length = None if hop_length_ms is None else int(hop_length_ms / 1000 * rate)
    # win_length = None if win_length_ms is None else int(win_length_ms / 1000 * rate)
    # S = librosa.stft(y=preemphasis_y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # spec_orig = np.abs(S)
    ####################################################
    ####################################################
    spec_orig = _amplitude_to_db(spec_orig)

    # Get useful stft analysis parameters
    hop_length_sec = hop_length_ms / 1000
    half_win_length_sec = win_length_ms / 1000

    # Cut spectrogram to spectral range
    if spectral_range is not None:
        spec_bin_hz = (rate / 2) / np.shape(spec_orig)[0]
        spec_orig = spec_orig[
            int(spectral_range[0] / spec_bin_hz):int(spectral_range[1] / spec_bin_hz),
            :,
        ]

    # loop through possible thresholding configurations starting at the highest
    for _, mldb in enumerate(
        tqdm(
            np.arange(_min_level_db(), min_level_db_floor, db_delta),
            leave=False,
            disable=(not verbose),
        )
    ):
        # set the minimum dB threshold
        # normalize the spectrogram
        spec = _normalize(spec_orig, min_db=mldb, max_db=ref_level_db)

        # subtract the median
        spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
        spec[spec < 0] = 0

        # get the vocal envelope
        vocal_envelope = np.max(spec, axis=0) * np.sqrt(np.mean(spec, axis=0))
        # normalize envelope
        vocal_envelope = vocal_envelope / np.max(vocal_envelope)

        ########################################
        ########################################
        # print vocal enveloppe
        plt.plot(vocal_envelope)
        plt.savefig(f'debug/vocal_enveloppe_{mldb}.pdf')
        plt.clf()
        ########################################
        ########################################

        # Get onsets and offsets both for voiced and silenced parts
        # We take the middle of the anlysis window as our onset/offset window (half_win_length_sec)
        onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold, time_bins)
        onsets_sil, offsets_sil = onsets_offsets(vocal_envelope <= silence_threshold, time_bins)

        # if there is a silence of at least min_silence_for_spec length,
        #  and a vocalization of no greater than max_vocal_for_spec length, the env is good
        if len(onsets_sil) > 0:
            # longest silences and periods of vocalization
            max_silence_len = np.max(offsets_sil - onsets_sil)
            max_vocalization_len = np.max(offsets - onsets)
            if verbose:
                print("longest silence", max_silence_len)
                print("longest vocalization", max_vocalization_len)

            if max_silence_len > min_silence_for_spec:
                if max_vocalization_len < max_vocal_for_spec:
                    envelope_is_good = True
                    break
        if verbose:
            print("Current min_level_db: {}".format(mldb))

    if not envelope_is_good:
        return None

    onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold, time_bins)

    # threshold out short syllables
    length_mask = (offsets - onsets) >= min_syllable_length_s

    return {
        "spec": spec,
        "vocal_envelope": vocal_envelope.astype("float32"),
        "min_level_db": mldb,
        "onsets": onsets[length_mask],
        "offsets": offsets[length_mask],
    }

def onsets_offsets(signal, time_bins):
    """
    [summary]

    Arguments:
        signal {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    elements, nelements = ndimage.label(signal)
    if nelements == 0:
        return np.array([[0], [0]])
    onsets_bin, offsets_bin = np.array(
        [
            np.where(elements == element)[
                0][np.array([0, -1])]  # + np.array([0, 1])
            for element in np.unique(elements)
            if element != 0
        ]
    ).T
    onsets = bins_to_seconds(onsets_bin, time_bins)
    offsets = bins_to_seconds(offsets_bin, time_bins)
    return np.array([onsets, offsets])


def bins_to_seconds(ll, time_bins):
    return [time_bins[e] for e in ll]

def plot_segmented_spec(
    spec, onsets, offsets, hop_length_ms, background="black", figsize=(30, 5)
):
    """ plot spectrogram with colormap labels
    """
    pal = np.random.permutation(sns.color_palette("hsv", n_colors=len(onsets)))
    fft_rate = 1000 / hop_length_ms
    new_spec = np.zeros(list(np.shape(spec)) + [4])
    for onset, offset, pi in zip(onsets, offsets, pal):
        if background == "black":
            cdict = {
                "red": [(0, pi[0], pi[0]), (1, 1, 1)],
                "green": [(0, pi[1], pi[1]), (1, 1, 1)],
                "blue": [(0, pi[2], pi[2]), (1, 1, 1)],
                "alpha": [(0, 0, 0), (0.25, 0.5, 0.5), (1, 1, 1)],
            }
        else:
            cdict = {
                "red": [(0, pi[0], pi[0]), (1, 0, 0)],
                "green": [(0, pi[1], pi[1]), (1, 0, 0)],
                "blue": [(0, pi[2], pi[2]), (1, 0, 0)],
                "alpha": [(0, 0, 0), (1, 1, 1)],
            }

        cmap = LinearSegmentedColormap("CustomMap", cdict)

        start_frame = int(onset * fft_rate)
        stop_frame = int(offset * fft_rate)
        new_spec[:, start_frame:stop_frame, :] = cmap(
            spec[:, start_frame:stop_frame])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(background)
    ax.imshow(new_spec, interpolation=None, aspect="auto", origin="lower")


def plot_segmentations(
    spec, vocal_envelope, onsets, offsets, hop_length_ms, rate, figsize=(30, 5)
):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    gs.update(hspace=0.0)  # set the spacing between axes.
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    plot_spec(spec, fig, ax1, rate=rate,
              hop_len_ms=hop_length_ms, show_cbar=False)
    ax0.plot(vocal_envelope, color="k")
    ax0.set_xlim([0, len(vocal_envelope)])
    ax1.xaxis.tick_bottom()
    ylmin, ylmax = ax1.get_ylim()
    ysize = (ylmax - ylmin) * 0.1
    ymin = ylmax - ysize

    patches = []
    for onset, offset in zip(onsets, offsets):
        ax1.axvline(onset, color="#FFFFFF", ls="dashed", lw=0.75)
        ax1.axvline(offset, color="#FFFFFF", ls="dashed", lw=0.75)
        patches.append(Rectangle(xy=(onset, ymin),
                                 width=offset - onset, height=ysize))

    collection = PatchCollection(patches, color="white", alpha=0.5)
    ax1.add_collection(collection)
    ax0.axis("off")
    return fig
