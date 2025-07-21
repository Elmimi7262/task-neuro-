import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

i1, i2 = 0.05, 0.10
target_o1, target_o2 = 0.01, 0.99
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
lr = 0.5

net_h1 = i1 * w1 + i2 * w3 + b1
net_h2 = i1 * w2 + i2 * w4 + b1
out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

net_o1 = out_h1 * w5 + out_h2 * w7 + b2
net_o2 = out_h1 * w6 + out_h2 * w8 + b2
out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

error_o1 = target_o1 - out_o1
error_o2 = target_o2 - out_o2

delta_o1 = error_o1 * sigmoid_derivative(out_o1)
delta_o2 = error_o2 * sigmoid_derivative(out_o2)

dw5 = lr * delta_o1 * out_h1
dw6 = lr * delta_o2 * out_h1
dw7 = lr * delta_o1 * out_h2
dw8 = lr * delta_o2 * out_h2

w5 += dw5
w6 += dw6
w7 += dw7
w8 += dw8

b2 += lr * (delta_o1 + delta_o2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * sigmoid_derivative(out_h1)
delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * sigmoid_derivative(out_h2)

dw1 = lr * delta_h1 * i1
dw2 = lr * delta_h2 * i1
dw3 = lr * delta_h1 * i2
dw4 = lr * delta_h2 * i2

w1 += dw1
w2 += dw2
w3 += dw3
w4 += dw4

b1 += lr * (delta_h1 + delta_h2)

print(f"{w1=:.4f}, {w2=:.4f}, {w3=:.4f}, {w4=:.4f}")
print(f"{w5=:.4f}, {w6=:.4f}, {w7=:.4f}, {w8=:.4f}")
print(f"{b1=:.4f}, {b2=:.4f}")
