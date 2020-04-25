from torch.utils.data import DataLoader

from avgn.pytorch.VAE import VAE


def get_model(model_type, model_kwargs, encoder, decoder, model_dir):
    if model_type == 'VAE':
        return VAE(encoder=encoder,
                   decoder=decoder,
                   beta=model_kwargs['beta'],
                   model_dir=model_dir)
    else:
        return None


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
