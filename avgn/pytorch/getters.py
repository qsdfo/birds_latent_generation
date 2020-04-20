from avgn import pytorch


def get_model(model_type, model_kwargs, encoder, decoder):
    if model_type == 'VAE':
        pytorch.VAE(model_kwargs=model_kwargs,
                    encoder=encoder,
                    decoder=decoder)