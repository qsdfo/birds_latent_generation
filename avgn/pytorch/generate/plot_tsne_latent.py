import numpy as np
from sklearn.manifold import TSNE
from avgn.utils.cuda_variable import cuda_variable
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tsne_latent(model, dataloader, savepath):
    # Forward pass
    model.eval()
    zs = []
    labels = []
    for batch_counter, data in enumerate(dataloader):
        x_cuda = cuda_variable(data['input'])
        # Get z
        mu, logvar = model.encode(x_cuda)
        z = model.reparameterize(mu, logvar) 
        zs += list(z.cpu().detach().numpy())
        labels += data['label']
        if batch_counter > 5:
            break
    z_embedded = TSNE(n_components=3).fit_transform(zs)
    
    label_to_points = {}
    for index, label in enumerate(labels):
        if label not in label_to_points:
            label_to_points[label] = []
        label_to_points[label].append(z_embedded[index])

    # Plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colormap = mpl.cm.tab20.colors
    for colorind, (label, points) in enumerate(label_to_points.items()):
        points_np = np.asarray(points)
        color = colormap[colorind]
        xs = points_np[:, 0]
        ys = points_np[:, 1]
        zs = points_np[:, 2]
        ax.scatter(xs, ys, zs, c=color, marker='o')
    plt.savefig(f'{savepath}/tsne.pdf')
    plt.close('all')