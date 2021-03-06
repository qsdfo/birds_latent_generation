from avgn.utils.hparams import HParams
from avgn.pytorch.decoder import Decoder
from avgn.pytorch.encoder import Encoder
from avgn.pytorch.spectro_dataset import SpectroDataset
import glob
import pickle
import random
from avgn.utils.paths import DATA_DIR
from datetime import datetime
import importlib
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from avgn.pytorch.VAE import VAE
from avgn.pytorch.VAE_categorical import VAE_categorical


def get_model(model_type, model_kwargs, encoder, decoder, model_dir):
    if model_type == 'VAE':
        return VAE(encoder=encoder,
                   decoder=decoder,
                   beta=model_kwargs['beta'],
                   model_dir=model_dir)
    if model_type == 'VAE_categorical':
        return VAE_categorical(encoder=encoder,
                               decoder=decoder,
                               beta=model_kwargs['beta'],
                               model_dir=model_dir)
    else:
        raise Exception


def get_dataloader(dataset_type, dataset, batch_size, shuffle):
    dataloader_ = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=0
                             )
    if dataset_type == 'mnist':
        # Remove label, don't need it here
        def dataloading(dataloader):
            for p in dataloader:
                yield p[0]

        dataloader = dataloading(dataloader_)
    else:
        dataloader = dataloader_
    return dataloader


def get_model_and_dataset(config, loading_epoch):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')
    if len(gpu_ids) == 0:
        device = 'cpu'
    else:
        device = 'cuda'

    # Load config
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config

    # compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        config['timestamp'] = timestamp

    if loading_epoch is not None:
        model_path = os.path.dirname(config_path)
    else:
        model_path = f'models/{config["savename"]}_{timestamp}'

    ##################################################################################
    print('##### Dataset')
    dataset_name = config['dataset']

    if dataset_name == 'mnist':
        dataset_train = datasets.MNIST('data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ]))
        dataset_val = datasets.MNIST('data', train=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             (0.1307,), (0.3081,))
                                     ]))
        hparams = None
    else:
        # Hparams
        hparams_loc = DATA_DIR / 'syllables' / \
            f'{dataset_name}_{config["dataset_preprocessing"]}_hparams.pkl'
        with open(hparams_loc, 'rb') as ff:
            hparams = pickle.load(ff)

        # FIXME: Legacy
        if not hasattr(hparams, 'pad_length'):
            hparams.pad_length = 64

        # data
        data_loc = DATA_DIR / 'syllables' / \
            f'{dataset_name}_{config["dataset_preprocessing"]}'
        syllable_paths = glob.glob(f'{str(data_loc)}/[0-9]*')
        random.seed(1234)
        random.shuffle(syllable_paths)
        num_syllables = len(syllable_paths)
        print(f'# {num_syllables} syllables')
        split = 0.9
        syllable_paths_train = syllable_paths[: int(split * num_syllables)]
        syllable_paths_val = syllable_paths[int(split * num_syllables):]
        dataset_train = SpectroDataset(syllable_paths_train)
        dataset_val = SpectroDataset(syllable_paths_val)
        # dataset_train = SpectroCategoricalDataset(syllable_df_train)
        # dataset_val = SpectroCategoricalDataset(syllable_df_val)

    ##################################################################################
    print('##### Build model')
    encoder_kwargs = config['encoder_kwargs']
    encoder = Encoder(
        n_z=config['n_z'],
        conv_stack=encoder_kwargs['conv_stack'],
        conv2z=encoder_kwargs['conv2z']
    )
    decoder_kwargs = config['decoder_kwargs']
    decoder = Decoder(
        deconv_input_shape=decoder_kwargs['deconv_input_shape'],
        z2deconv=decoder_kwargs['z2deconv'],
        deconv_stack=decoder_kwargs['deconv_stack'],
        n_z=config['n_z']
    )
    model = get_model(
        model_type=config['model_type'],
        model_kwargs=config['model_kwargs'],
        encoder=encoder,
        decoder=decoder,
        model_dir=model_path
    )
    optimizer = torch.optim.Adam(list(model.parameters()), lr=config['lr'])

    if loading_epoch is not None:
        print('##### Load model')
        model.load(name=loading_epoch, device=device)

    model.to(device)
    return model, dataset_train, dataset_val, optimizer, hparams, config, model_path, config_path
