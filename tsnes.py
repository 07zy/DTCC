from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne(dataset,x, y):
    x = TSNE().fit_transform(x)
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=6, marker='o')
    plt.savefig('./Figure/{}.pdf'.format(dataset))
    plt.show()
    plt.close()

def middle_tsne(dataset,x, y):
    x = TSNE().fit_transform(x)
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=6, marker='o')
    plt.savefig('./Figure/{}_epoch=50.pdf'.format(dataset))
    plt.show()
    plt.close()
