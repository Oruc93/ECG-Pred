import numpy as np
import matplotlib.pyplot as plt

X_test = np.load("./X_test.npy")
y_test = np.load("./y_test.npy")
y_pred = np.load("./y_pred.npy")
y_pred = y_pred[:,:,:,0]
test = y_pred.reshape(np.shape(y_test))

test1 = np.ones(np.shape(y_test))
test1[:,:,0] = y_pred[0,]

# print(np.shape(X_test))
# print(np.shape(y_test))
print(np.shape(y_pred))
print(np.shape(test))
print(np.sum(test[0,:,0]==y_pred[0,0,:]))
print(10000)

plt.figure(1)
plt.title("Full plot Truth and Pred of lag 1")
plt.plot(list(range(len(y_test[0,:,0]))), y_test[0,:,0])
plt.plot(list(range(len(y_pred[0,0,:]))), y_pred[0,0,:])
plt.savefig("Full-Plot lag 1")

"""for k in range(len(y_pred[0,0,:])):
    plt.figure(k+2)
    plt.title("Zoomed Truth and Pred of column " + str(k))
    plt.plot(list(range(len(y_test[0,:,0]))), y_test[0,:,0])
    plt.plot(list(range(len(y_pred[0,:,0]))), y_pred[0,:,0])
    name = "./ZoomPlot-Col-" + str(k)
    plt.savefig(name)"""