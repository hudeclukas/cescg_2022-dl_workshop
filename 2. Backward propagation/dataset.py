import numpy as np
import plotly.express as px

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

def draw_dataset(x, y):
    if x.shape[1] == 2:
        fig = px.scatter(x=x[:,0,0], y=x[:,1,0], color=y[:,0,0], width=500, height=500, color_continuous_scale='Bluered')
    else:
        return
    fig.show()


if __name__ == '__main__':
    x,y = dataset_circles(128)
    print(x.shape)
    print(y.shape)
    draw_dataset(x, y)