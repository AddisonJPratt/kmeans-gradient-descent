# %%
# Gradient Descent for linear regression
# y = betax + beta0
# loss = (y-yhat)**2 / N

# Initialise some parameters
#create gradient descent function
import numpy as np

# %%
x = np.random.randn(10,1)
y = 2*x + np.random.rand()
# hyperparameters
learning_rate = 0.01

# parameters
w = 0.0
b = 0.0

def descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    # loss = (y-(wx+b))))**2
    for xi, yi in zip(x,y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb

    return(w,b)

# %%
for epoch in range(800):
    # run gradient descent here
    w, b = descend(x, y, w, b, learning_rate)
    yhat = w*x + b
    loss =np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])

    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')
    pass
# %%
