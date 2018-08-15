import glob
import numpy as np
import time
from scipy.misc import imsave, imread, imshow

if __name__ == "__main__":
    start = time.process_time()
    target_folder = "training_data"
    vid_folder = glob.glob("{}/*.png".format(target_folder))
    n = len(vid_folder)
    training_data = []
    for i in range(1, n+1):
        image_path = "training_data/training_data" + str(i) + ".png"
        temp = np.array(imread(image_path), dtype=np.uint8)/255
        temp = np.swapaxes(temp, 0,1)
        temp = np.reshape(temp, (len(temp)*len(temp[0])))
        training_data.append(temp)

    average = np.sum(training_data, 0)/n
    average = np.reshape(average, (1, len(average)))
    training_data = np.subtract(training_data, average*np.ones((n, 1)))
    cov = np.dot(training_data, training_data.T)
    [V, D] = np.linalg.eig(cov)
    V = np.abs(V)
    V = np.sqrt(V)
    V = V*np.eye(n)
    V = np.linalg.inv(V)
    D = np.swapaxes(D, 0, 1)
    #Correct Up to here
    U = np.dot(D, training_data)
    U = np.dot(V, U)
    print(U[0][0:10])
    end = time.process_time()
    U[0] = 255 * U[0]+255*average
    x = np.vstack((np.hsplit(U[0], 640)))
    x = np.swapaxes(x, 0, 1)
    imsave("test.png", x)
    print("Total Time = {}".format(end-start))


