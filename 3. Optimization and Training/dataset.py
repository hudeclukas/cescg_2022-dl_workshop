import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

def dataset_circles(m=10, radius=0.7, noise=0.0):
    # Element values are in the interval <-1; 1>
    X = (np.random.rand(m, 2, 1) * 2.0) - 1.0

    # Element-wise multiplication with random noise
    N = (np.random.rand(m, 2, 1) - 0.5) * noise
    Xnoise = X + N

    # Compute the radius
    # Element-wise square
    XSquare = Xnoise ** 2

    # Sum over axis=1. We get a (m, 1) array.
    RSquare = np.sum(XSquare, axis=1, keepdims=True)
    R = np.sqrt(RSquare)

    # Y is 1, if radius `R` is greater than `radius`
    Y = (R > radius).astype(float)

    # Return X, Y
    return X, Y

def dataset_Flower(m=10, noise=0.0):
    # Init matrices
    X = np.zeros((m, 2), dtype='float')
    Y = np.zeros((m, 1), dtype='float')

    a = 1.0
    pi = 3.141592654
    M = int(m / 2)

    for j in range(2):
        ix = range(M * j, M * (j + 1))
        t = np.linspace(j * pi, (j + 1) * pi, M) + np.random.randn(M) * noise
        r = a * np.sin(4 * t) + np.random.randn(M) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y

def MakeBatches(dataset, batchSize, shuffle:bool):
    # Dataset contain 2 sets - X, Y
    X, Y = dataset

    # Get the total number of samples
    m, nx = X.shape
    _, ny = Y.shape

    if shuffle:
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

    result = []

    # If batchSize = 0, let's take the whole set
    if (batchSize <= 0):
        batchSize = m

    # Total number of mini-batches is rounded up
    steps = int(np.ceil(m / batchSize))
    for i in range(steps):
        # count cut intervals
        mStart = i * batchSize
        mEnd = min(mStart + batchSize, m)

        # Select samples for actual cut + we need to preserve the rank [B, F, 1]
        minibatchX = X[mStart:mEnd, :]
        minibatchY = Y[mStart:mEnd, :]

        assert (len(minibatchX.shape) == 2)
        assert (len(minibatchY.shape) == 2)

        # Append new miniBatch to result
        result.append((np.expand_dims(minibatchX, axis=-1), np.expand_dims(minibatchY, axis=-1)))

    return result


def draw_dataset(x, y):
    fig = px.scatter(x=x[0], y=x[1], color=y[0], width=700, height=700)
    fig.show()


def draw_DecisionBoundary(X, Y, model):
    # FInd the borders to which we want to analyze the predictions
    pad = 0.5
    x1_Min, x1_Max = X[0, :].min() - pad, X[0, :].max() + pad
    x2_Min, x2_Max = X[1, :].min() - pad, X[1, :].max() + pad

    # Make a grid of samples and we subsample from the whole interval <MIN; MAX> with granularity h
    h = 0.01
    x1_Grid, x2_Grid = np.meshgrid(
        np.arange(x1_Min, x1_Max, h),
        np.arange(x2_Min, x2_Max, h)
    )

    # Order the value grid to the same shape as X
    XX = np.c_[x1_Grid.ravel(), x2_Grid.ravel()].T

    # Vypocitame predikciu pomocou modelu na vsetky hodnoty mriezky
    YHat = model(XX)

    # A usporiadame si vysledok tak, aby sme ho mohli podhodit PyPlotu
    YHat = YHat.reshape(x1_Grid.shape)

    # Najskor nakreslime contour graf - vysledky skumania pre mriezku
    plt.figure(figsize=(9, 9))
    plt.xscale('linear')
    plt.yscale('linear')
    plt.contourf(x1_Grid, x2_Grid, YHat, cmap=plt.cm.RdYlBu)

    # Potom este pridame scatter graf pre X, Y
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.RdBu)

    plt.show()
    plt.close()

if __name__ == '__main__':
    x,y = dataset_Flower(128)
    draw_dataset(x.T, y.T)
    dataset = MakeBatches((x,y),32,True)
    for mini_batch in dataset:
        print(mini_batch[0].shape)
