import matplotlib.pyplot as plt
from src.gen_manifold import manifoldGen
import sys
from src.smce import smce
import numpy as np

def eval_2refoils(verbose = True):
    Y, x, gtruth, _ = manifoldGen('2trefoils')
    lambda_val = 10
    KMax = 50
    dim = 2
    n = len(np.unique(gtruth))
    
    # Run SMCE algorithm
    Yc, Yj, clusters, missrate = smce(Y, lambda_val, KMax, dim, n, gtruth, verbose)
    print(f'Missrate: {missrate}')

    # viusualize the results
    plt_2refoils(x, Yc, clusters)

def eval_sphere(verbose = True):
    # Embedding of 2D Sphere via SMCE
    Y, x, gtruth, _ = manifoldGen('sphere')
    lambda_val = 10
    KMax = 50
    dim = 2
    n = len(np.unique(gtruth))

    # Run SMCE algorithm
    Yc, Yj, clusters, missrate = smce(Y, lambda_val, KMax, dim, n, gtruth, verbose)
    print(f'Missrate: {missrate}')

    # viusualize the results
    plt_shpere(x, Yc)

def plt_2refoils(x, x_smce, clusters):
    N = x.shape[1]
    n = len(np.unique(clusters))
    # Plot the original data points
    fig = plt.figure(figsize = (12, 4))
    ax = fig.add_subplot(131, projection='3d')
    colorr = plt.get_cmap('jet')(np.linspace(0, 1, N))
    for j in range(N):
        ax.scatter(x[0, j], x[1, j], x[2, j], color=colorr[j], s=9)
    ax.set_title('Trefoil-knots embedded in R^100', fontsize=16)

    # Plot the embedding(s)
    f = 1
    for i in range(n):
        color_i = plt.get_cmap('jet')(np.linspace(0, 1, N))[clusters == i]
        f += 1
        ax = fig.add_subplot(130 + f)
        cluster_embed = x_smce[i]
        for j in range(cluster_embed.shape[1]):
            ax.scatter(cluster_embed[0, j], cluster_embed[1, j], color=color_i[j], s=9)
        ax.set_title(f'Embedding of cluster {i}', fontsize=16)
        ax.axis('equal')
    plt.show()
    
def plt_shpere(x, x_smce):
    N = x.shape[1]
    # Plot the original data points
    fig = plt.figure(figsize=(12, 6))

    # Plot original data points
    ax1 = fig.add_subplot(121, projection='3d')
    colorr = plt.cm.jet(np.arange(N) / N)
    for j in range(N):
        ax1.scatter(x[0, j], x[1, j], x[2, j], color=colorr[j], s=90)
    ax1.set_title('2D Sphere embedded in R^100', fontsize=16)
    ax1.set_aspect('auto')

    # Plot the embedding of the sphere
    ax2 = fig.add_subplot(122)
    color = plt.cm.jet(np.arange(N) / N)

    cluster_embed = x_smce[0]
    for j in range(cluster_embed.shape[1]):
        ax2.scatter(cluster_embed[0, j], cluster_embed[1, j], color=color[j], s=90)
    ax2.set_title('Embedding of the sphere', fontsize=16)
    ax2.set_aspect('equal')
    plt.show()

if __name__ == '__main__':
    data_set=sys.argv[1]
    if data_set=="2refoils":
        eval_2refoils()
    elif data_set=="sphere":
        eval_sphere()
    else:
        print("Please specify a valid dataset: 2refoils or sphere")