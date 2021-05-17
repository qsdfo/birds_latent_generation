from avgn.signalprocessing.spectrogramming_scipy import build_mel_basis, build_mel_inversion_basis, inv_spectrogram_sp
import matplotlib.pyplot as plt
import soundfile as sf


def plot_generation(model, data_processing, num_examples, savepath):
    # forward pass
    model.eval()
    gen = model.generate(batch_dim=num_examples).cpu().detach().numpy()

    # plot
    dims = gen.shape[2:]
    plt.clf()
    fig, axes = plt.subplots(ncols=num_examples)
    for i in range(num_examples):
        # show the image
        axes[i].matshow(gen[i].reshape(dims), origin="lower")
    for ax in fig.get_axes():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(f'{savepath}/spectro.pdf')
    plt.close('all')

    # audio
    audios = []
    mel_basis = build_mel_basis(data_processing['n_fft'], data_processing['sr'], data_processing['num_mel_bins'],
                                data_processing['mel_lower_edge_hertz'], data_processing['mel_upper_edge_hertz'])
    mel_inversion_basis = build_mel_inversion_basis(mel_basis)
    for i in range(num_examples):
        gen_audio = inv_spectrogram_sp(gen[i, 0],
                                       n_fft=data_processing['n_fft'],
                                       win_length=data_processing['win_length'],
                                       hop_length=data_processing['hop_length'],
                                       ref_level_db=data_processing['ref_level_db'],
                                       power=data_processing['power'],
                                       mel_inversion_basis=mel_inversion_basis
                                       )
        audios.append(gen_audio)
        sf.write(f'{savepath}/{i}.wav', gen_audio,
                 samplerate=data_processing['sr'])
    return {
        'audios': audios,
        'spectros': gen
    }
