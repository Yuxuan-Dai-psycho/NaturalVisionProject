# This file contains the taget roi and class selection.
# Outlier detection and other preprocessing method on data



#=========Function of roi and class selection=========
def select_roi():
    pass

def select_class():
    pass



#%% =========Function of Outlier detection=========
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

#%%
def plot_outlier(data, cont):
    # input: data,  contamination -> outlier ratio
    # output: scatter figure;
    # return: X_pca -> PCA processed data, array of float64
    #         y_pred -> outlier mark, array of int64 (1 & -1)
    
    # scaler
    scaler = StandardScaler()
    scaler.fit(data)
    X_scaled = scaler.transform(data)
    
    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)
    
    # EllipticEnvelope
    esti = EllipticEnvelope(contamination=cont)
    y_pred = esti.fit(X_pca).predict(X_pca)
    
    # contour
    X,Y = np.meshgrid(np.linspace(-120,120,500),
                      np.linspace(-30,30,200))
    Z = esti.predict(np.c_[X.ravel(), Y.ravel()])
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z, levels=[0], colors='black')
    
    # plot
    colors = np.array(['#377eb8', '#ff7f00'])
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='.', color=colors[(y_pred+1)//2])
    plt.show()
    
    # return
    return X_pca, y_pred

#%%
def remove_outlier(data_raw, out_mark):
    # input: data_raw, outmark -> outlier mark, array of int64 (1 & -1)
    # output: data_pro -> processed data
    
    # outlier index
    out_index = np.where(out_mark == -1)
    out_index = out_index[0]
    
    # numpy delete
    data_pro = np.delete(data_raw, out_index, axis=0)
    
    # return
    return data_pro