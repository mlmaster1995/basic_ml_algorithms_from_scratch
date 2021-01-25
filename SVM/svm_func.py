import random
import numpy as np
import matplotlib.pyplot as plt


## generate random weight w0,w1,w2,w3.....
def generate_weight(num_attr, start, end):
    wt_array = []
    for i in range(num_attr):
        wt_array.append(random.uniform(start, end))
    return wt_array


## SVM_test with online learning
def SVM_gaussian_test(dt_test, dt_test_target, wt_arr, online, stp, C):
    error = 0
    update = 0
    for index in range(len(dt_test)):
        r = np.dot(dt_test[index], wt_arr) * dt_test_target[index]
        if r < 0:
            error += 1

            if online:
                update += 1
                wt_arr = wt_arr + stp * (dt_test_target[index] * np.array(dt_test[index]) - (1/C) * wt_arr)

    print("online training update:", update, '\n')
    score = 1-error / len(dt_test)
    return score


## gaussian_kernel func
def gaussian_kernel(x_ipt, lama_single, sigma):
    prob = np.exp(-(np.linalg.norm(abs(x_ipt-lama_single)))**2/(2*sigma**2))
    return prob


## generate feature func
def gaussian_feature_make(dt_train, lama, sigma, round_limit=10):
    feature=[]
    for val in dt_train:
        feature.append([round(gaussian_kernel(val, sig, sigma), round_limit) for sig in lama])
        feature[-1].insert(0, 1)
    return feature


## polynomial_kernel func
def poly_kernel(x_ipt, lama_single, degree=2, bias=0):
    poly_val = (np.dot(x_ipt, lama_single)+bias)**degree
    return poly_val


## generate poly features
def poly_feature_make(dt_train, lama, degree=2, bias=0):
    feature=[]
    for val in dt_train:
        feature.append([poly_kernel(val, sig, degree, bias) for sig in lama])
        feature[-1].insert(0, 1)
    return feature


## SVM func
def SVM(dt_train, dt_train_target, wt_arr, kernel='linear', C=1, sigma=0.5, degree=2, bias=0, stp=1, epoch_limit=2000,
         stp_limit=1e-300, stp_show=False, plot=False):

    epoch = 0
    hinge_loss_all = 0
    reg_val = 1 / C
    lama = np.array(dt_train)

    epoch_plt = []
    hinge_loss_plt = []


    if kernel == 'gaussian':
        features = gaussian_feature_make(dt_train, lama, sigma, round_limit=5)


    while epoch <= epoch_limit:
        hinge_loss = []
        epoch_plt.append(epoch)
        hinge_loss_last = hinge_loss_all

        for index in range(len(features)):

            r = np.dot(features[index], wt_arr) * dt_train_target[index]

            if r >= 1:
                hinge_loss.append(0)
                wt_arr = wt_arr - stp * reg_val * wt_arr
            else:
                hinge_loss.append(1 - r)
                wt_arr = wt_arr + stp * (dt_train_target[index] * np.array(features[index]) - reg_val * wt_arr)
            hinge_loss_all = sum(hinge_loss)

        hinge_loss_plt.append(hinge_loss_all)

        if abs(hinge_loss_last - hinge_loss_all) <= 0.1:
            stp = stp * 0.1

        if stp <= stp_limit:
            break

        epoch += 1

    if plot:
        # plot hinge loss
        plt.plot(epoch_plt, hinge_loss_plt)
        plt.title('hinge loss vs epochs')
        plt.xlabel('epochs')
        plt.ylabel('hingle loss')
        plt.grid()
        plt.show()

    return (wt_arr, C, stp)





