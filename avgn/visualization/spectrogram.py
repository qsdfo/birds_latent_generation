import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from tqdm.autonotebook import tqdm


def plot_example_specs(
    specs,
    labels,
    clusters_to_viz,
    custom_pal=sns.color_palette(),
    nex=4,
    line_width=10,
    ax=None,
    pad=1,
    figsize=(10, 10),
):
    spec_x = np.shape(specs[0])[0]
    spec_y = np.shape(specs[0])[1]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    canvas = np.zeros((spec_x * nex, spec_y * len(clusters_to_viz)))
    for ci, cluster in enumerate(clusters_to_viz):
        color = np.array(sns.color_palette(custom_pal, len(np.unique(labels))))[
            np.unique(labels) == cluster
        ][0]
        ex_specs = np.random.permutation(specs[labels == cluster])[:nex]
        for exi, ex in enumerate(ex_specs):
            canvas[
                exi * spec_x : (exi + 1) * spec_x, ci * spec_y : (ci + 1) * spec_y
            ] = ex / np.max(ex)

            rect = Rectangle(
                xy=[ci * spec_y - 0.5, (exi - 1) * spec_x - 0.5],
                width=spec_y * exi,
                height=line_width,
                linewidth=0,
                facecolor=color,
            )
            ax.add_patch(rect)

            rect = Rectangle(
                xy=[ci * spec_y - 0.5, exi * spec_x - 0.5],
                width=spec_y * exi,
                height=line_width,
                linewidth=0,
                facecolor=color,
            )
            ax.add_patch(rect)

        rect = Rectangle(
            xy=[ci * spec_y - 0.5, (exi + 1) * spec_x - 0.5 - line_width],
            width=spec_y * exi,
            height=line_width,
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(rect)

        rect = Rectangle(
            xy=[ci * spec_y - 0.5, 0 - 0.5],
            width=line_width,
            height=spec_x * len(ex_specs),
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(rect)

        rect = Rectangle(
            xy=[(ci + 1) * spec_y - line_width - 0.5, 0 - 0.5],
            width=line_width,
            height=spec_x * len(ex_specs),
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(rect)

    ax.matshow(
        canvas, origin="lower", interpolation="none", aspect="auto", cmap=plt.cm.bone
    )
    ax.axis("off")
    return ax


def plot_spec(spec, fig, ax, extent=None, cmap=plt.cm.afmhot, show_cbar=True):
    """plot spectrogram
    
    [description]
    
    Arguments:
        spec {[type]} -- [description]
        fig {[type]} -- [description]
        ax {[type]} -- [description]
    
    Keyword Arguments:
        cmap {[type]} -- [description] (default: {plt.cm.afmhot})
    """
    spec_ax = ax.matshow(
        spec,
        interpolation=None,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=extent,
    )
    if show_cbar:
        cbar = fig.colorbar(spec_ax, ax=ax)
        return spec_ax, cbar
    else:
        return spec_ax


def visualize_spec(
    wav_spectrogram,
    save_loc=None,
    show=True,
    figsize=(20, 5),
    extent=None,
    cmap=plt.cm.afmhot,
    show_cbar=True,
):
    """basic spectrogram visualization and saving
    
    [description]
    
    Arguments:
        wav_spectrogram {[type]} -- [description]
    
    Keyword Arguments:
        save_loc {[type]} -- [description] (default: {None})
        show {bool} -- [description] (default: {True})
        figsize {tuple} -- [description] (default: {(20,5)})
    """
    fig, ax = plt.subplots(figsize=figsize)
    spec_ax, cbar = plot_spec(wav_spectrogram, fig, ax)
    if show:
        plt.show()
    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches="tight")
        plt.close()


def plot_segmentations(
    spec,
    vocal_envelope,
    all_syllable_starts,
    all_syllable_lens,
    fft_rate,
    hparams,
    fig=None,
    axs=None,
    figsize=(60, 9),
):
    """Plot the segmentation points over a spectrogram
    
    [description]
    
    Arguments:
        spec {[type]} -- [description]
        vocal_envelope {[type]} -- [description]
        all_syllable_start {[type]} -- [description]
        all_syllable_lens {[type]} -- [description]
        fft_rate {[type]} -- [description]
    
    Keyword Arguments:
        figsize {tuple} -- [description] (default: {(60, 9)})
    """
    stop_time = np.shape(spec)[1] / fft_rate
    if fig is None and axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=figsize)
    extent = [0, stop_time, 0, hparams["sample_rate"] / 2]
    plot_spec(spec, fig, axs[0], extent=extent, show_cbar=False)

    # plot segmentation marks
    for st, slen in zip(all_syllable_starts, all_syllable_lens):
        axs[0].axvline(st, color="w", linestyle="-", lw=3, alpha=0.75)
        axs[0].axvline(st + slen, color="w", linestyle="-", lw=3, alpha=0.75)

    axs[1].plot(vocal_envelope, color="k", lw=4)
    axs[1].set_xlim([0, len(vocal_envelope)])

    plt.show()


def draw_spec_set(spectrograms, maxrows=3, colsize=10, cmap=plt.cm.afmhot, zoom=2):
    """
    """
    # get column and row sizes
    rowsize = np.shape(spectrograms[0])[0]
    colsize = colsize * rowsize

    # create the vanvas
    canvas = np.zeros((rowsize * maxrows, colsize))

    # fill the canvas
    column_pos = 0
    row = 0
    for speci, spec in tqdm(enumerate(spectrograms)):
        spec_shape = np.shape(spec)
        if column_pos + spec_shape[1] > colsize:
            if row == maxrows - 1:
                break
            row += 1
            column_pos = 0
        canvas[
            rowsize * (maxrows-1-row) : rowsize * ((maxrows-1-row) + 1), column_pos : column_pos + spec_shape[1]
        ] = spec
        column_pos += spec_shape[1]
    if row < maxrows - 1:
        canvas = canvas[(maxrows-1-row) * rowsize:, :]
    #print(speci)
    figsize = (zoom * (colsize / rowsize), zoom * (row + 1))
    # print(figsize, np.shape(canvas), colsize / rowsize, rowsize, colsize)
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(
        canvas, cmap=cmap, origin="lower", aspect="auto", interpolation="nearest"
    )
    ax.axis("off")
    plt.show()


def plot_syllable_list(
    all_syllables,
    n_mel_freq_components,
    max_rows=3,
    max_sylls=100,
    width=400,
    zoom=1,
    spacing=1,
    cmap=plt.cm.viridis,
):
    """Plot a list of syllables as one large canvas
    
    [description]
    
    Arguments:
        all_syllables {[type]} -- [description]
        hparams {[type]} -- [description]
    
    Keyword Arguments:
        max_rows {number} -- [description] (default: {3})
        max_sylls {number} -- [description] (default: {100})
        width {number} -- [description] (default: {400})
        zoom {number} -- [description] (default: {1})
        spacing {number} -- [description] (default: {1})
        cmap {[type]} -- [description] (default: {plt.cm.viridis})
    """
    canvas = np.zeros((n_mel_freq_components * max_rows, width))
    x_loc = 0
    row = 0

    for i, syll in enumerate(all_syllables):

        # if the syllable is too long
        if np.shape(syll)[1] > width:
            continue

        if (x_loc + np.shape(syll)[1]) > width:
            if row == max_rows - 1:
                break

            else:
                row += 1
                x_loc = 0

        canvas[
            row * n_mel_freq_components : (row + 1) * n_mel_freq_components,
            x_loc : (x_loc + np.shape(syll)[1]),
        ] = np.flipud(syll)

        x_loc += np.shape(syll)[1] + spacing

    if row < max_rows:
        canvas = canvas[: (row + 1) * n_mel_freq_components]

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(width / 32 * zoom, max_rows * zoom)
    )
    ax.matshow(
        canvas,
        cmap=cmap,
        # origin='lower',
        aspect="auto",
        interpolation="nearest",
    )
    plt.show()


def plot_bout_to_syllable_pipeline(
    data,
    vocal_envelope,
    wav_spectrogram,
    all_syllables,
    all_syllable_starts,
    all_syllable_lens,
    rate,
    fft_rate,
    zoom=1,
    submode=True,
    figsize=(50, 10),
):
    """plots the whole plot_bout_to_syllable_pipeline pipeline
    """
    # create a plot where the top is waveform, underneath is spectrogram, underneath is segmented syllables
    # fig = plt.subplots(figsize=figsize)
    # gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,2,1])
    # ax=[plt.subplot(i) for i in gs]
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    # plot the original vocalization data
    ax[0].plot(data, color="black")
    ax[0].set_xlim([0, len(data)])
    ax[0].axis("off")
    ax[0].set_ylim([np.min(data), np.max(data)])

    # plot the vocal envelope below the data
    ax[1].plot(vocal_envelope)
    ax[1].set_xlim([0, len(vocal_envelope)])
    ax[1].set_ylim([np.min(vocal_envelope), np.max(vocal_envelope)])
    ax[1].axis("off")

    stop_time = np.shape(wav_spectrogram)[1] / fft_rate
    extent = [0, stop_time, 0, rate / 2]
    plot_spec(wav_spectrogram, fig, ax[2], extent=extent, show_cbar=False)

    # plot segmentation marks
    for st, slen in zip(all_syllable_starts, all_syllable_lens):
        ax[2].axvline(st, color="w", linestyle="-", lw=3, alpha=0.75)
        ax[2].axvline(st + slen, color="w", linestyle="-", lw=3, alpha=0.75)

    """    for si, syll_se in enumerate([(i[0], i[-1]) for i in all_syllables_time_idx]):
        imscatter(
            (syll_se[1] + syll_se[0]) / 2,
            0,
            np.flipud(norm(all_syllables[si])),
            zoom=zoom,
            ax=ax[3],
        )
        ax[3].text(
            (syll_se[1] + syll_se[0]) / 2,
            0.15,
            round(syllable_lengths[si], 3),
            fontsize=15,
            horizontalalignment="center",
        )"""

    ax[3].set_xlim([0, len(data)])
    ax[3].set_ylim([-0.2, 0.2])
    ax[3].axis("off")
    plt.tight_layout()
    plt.show()
