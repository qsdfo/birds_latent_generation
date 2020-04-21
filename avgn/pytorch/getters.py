from avgn.pytorch.VAE import VAE


def get_model(model_type, model_kwargs, encoder, decoder):
    if model_type == 'VAE':
        return VAE(encoder=encoder,
                   decoder=decoder)
    else:
        return None
