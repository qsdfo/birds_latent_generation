# from avgn.utils.cuda_variable import cuda_variable


# def latent_space_stats_per_species(model, dataloader, savepath):
#     # Forward pass
#     model.eval()
#     batch_counter = 0
#     mean = None
#     std = None
#     for data in dataloader:
#         x_cuda = cuda_variable(data['input'])
#         # Get z
#         mu, logvar = model.encode(x_cuda)
#         z = model.reparameterize(mu, logvar)
#         batch_dim, latent_dim = z.shape
#         if mean is None:
#         if batch_counter * batch_dim > 50000:
#             break

#     # Plot
#     dims = h_dim, w_dim
#     plt.clf()
#     fig, axes = plt.subplots(nrows=num_examples, ncols=num_interpolated_points)
#     for ind_example in range(num_examples):
#         for ind_interp in range(num_interpolated_points):
#             # show the image
#             axes[ind_example, ind_interp].matshow(x_interpolation[ind_example, :, :, :, ind_interp].reshape(dims),
#                                                   origin="lower")
#     for ax in fig.get_axes():
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.savefig(f'{savepath}/spectro.pdf')
#     plt.close('all')