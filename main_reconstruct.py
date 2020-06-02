import pickle
import click
import librosa
import matplotlib.pyplot as plt
import pandas as pd

from avgn.signalprocessing.spectrogramming import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_librosa
from avgn.utils.paths import DATA_DIR


@click.command()
@click.option('-d', '--dataset_name', type=str)
def main(dataset_name):

    ##################################################################################
    print(f'##### Dataset')

    # Choose a preprocessing pipeline
    # win_length_ms = None
    # hop_length_ms = 10
    # n_fft = 2048
    # num_mel_bins = 128
    # mel_lower_edge_hertz = 500
    # mel_upper_edge_hertz = 12000
    # suffix_preprocessing = f'wl{win_length_ms}_' \
    #                         f'hl{hop_length_ms}_' \
    #                         f'nfft{n_fft}_' \
    #                         f'melb{num_mel_bins}_' \
    #                         f'mell{mel_lower_edge_hertz}_' \
    #                         f'melh{mel_upper_edge_hertz}'
    suffix_preprocessing = 'wlNone_hl5_nfft4096_melb256_mell500_melh20000'

    df_loc = DATA_DIR / 'syllable_dfs' / dataset_name / f'data_{suffix_preprocessing}.pickle'
    syllable_df = pd.read_pickle(df_loc)

    hparams_loc = DATA_DIR / 'syllable_dfs' / dataset_name / f'hparams_{suffix_preprocessing}.pickle'
    with open(hparams_loc, 'rb') as ff:
        hparams = pickle.load(ff)

    dump_folder = DATA_DIR / 'dump' / 'reconstruction'
    for ind in range(len(syllable_df)):
        if ind > 50:
            break
        spec = syllable_df.spectrogram[ind] / 255.
        audio = syllable_df.audio[ind]

        # plt spectrogram
        plt.clf()
        plt.matshow(spec, origin="lower")
        plt.savefig(f'{dump_folder}/{ind}_spectro.pdf')
        plt.close()

        # melSpec to audio
        mel_basis = build_mel_basis(hparams, hparams.sr, hparams.sr)
        mel_inversion_basis = build_mel_inversion_basis(mel_basis)
        audio_reconstruct = inv_spectrogram_librosa(spec, hparams.sr, hparams, mel_inversion_basis=mel_inversion_basis)

        # save and plot audio original and reconstructed
        librosa.output.write_wav(f'{dump_folder}/{ind}_reconstruct.wav', audio_reconstruct, sr=hparams.sr, norm=True)
        librosa.output.write_wav(f'{dump_folder}/{ind}_original.wav', audio, sr=hparams.sr, norm=True)


def plot_reconstruction(model, dataloader, device, savepath):
    # Forward pass
    model.eval()
    for _, data in enumerate(dataloader):
        data_cuda = data.to(device)
        x_recon = model.reconstruct(data_cuda).cpu().detach().numpy()
        break
    # Plot
    dims = x_recon.shape[2:]
    num_examples = x_recon.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[0, i].matshow(data[i].reshape(dims), origin="lower")
        axes[1, i].matshow(x_recon[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(savepath)
    plt.close()
    plt.clf()


def plot_generation(model, num_examples, savepath):
    # forward pass
    model.eval()
    gen = model.generate(batch_dim=num_examples).cpu().detach().numpy()
    # plot
    dims = gen.shape[2:]
    fig, axes = plt.subplots(ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[i].matshow(gen[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(savepath)
    plt.close()
    plt.clf()


def epoch(model, optimizer, dataloader, num_batches, training, device):
    if training:
        model.train()
    else:
        model.eval()
    losses = []
    for batch_idx, data in enumerate(dataloader):
        if num_batches is not None and batch_idx > num_batches:
            break
        data_cuda = data.to(device)
        optimizer.zero_grad()
        loss = model.step(data_cuda)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss


if __name__ == '__main__':
    main()
