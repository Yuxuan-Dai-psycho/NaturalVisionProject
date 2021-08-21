import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join as pjoin
from scipy.spatial import distance_matrix

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(confusion, class_name, specify):
    """

    Parameters
    ----------
    specify : str
        Name to specify this plot

    """
    out_path = '/nfs/m1/BrainImageNet/Analysis_results/imagenet_decoding/results/confusion_matrix'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # get confusion
    n_class = confusion.shape[0]
    # visualize
    cmap = plt.cm.jet
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    font = {'family': 'serif', 'weight': 'bold', 'size':14}

    plt.imshow(confusion, cmap=cmap, norm=norm)
    plt.colorbar()
    
    plt.xlabel('Predict label', font)
    plt.ylabel('True label', font)
    plt.xticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), class_name.astype(int),
                fontproperties='arial', weight='bold', size=10)
    plt.yticks(np.linspace(0, n_class-1, n_class, dtype=np.uint8), class_name.astype(int),
                fontproperties='arial', weight='bold', size=10)
    plt.title(f'Confusion matrix {specify}', font)
    plt.savefig(pjoin(out_path, f'confusion_{specify}.jpg'))
    plt.close()


def save_classification_report(y_test, y_pred, specify):
    """

    Parameters
    ----------
    y_test : ndarray
        Groundtruth class 
    y_pred : ndarray
        Class predicted by model
    specify : str
        Name to specify this plot

    """
    out_path = '/nfs/m1/BrainImageNet/Analysis_results/imagenet_decoding/results/classification_report'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # generate report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(pjoin(out_path, f'classification_report_{specify}.csv'))

def top_k_acc(X_probs, y_test, k):
    """
    Accuracy on top k

    Parameters
    ----------
    X_probs : ndarray
        DESCRIPTION.
    y_test : ndarray
        GroundTruth
    k : int
        DESCRIPTION.

    Returns
    -------
    acc_top_k : TYPE
        DESCRIPTION.

    """
    # top k
    class_names = np.unique(y_test)
    best_n = np.argsort(X_probs, axis=1)[:, -k:]
    y_top_k = class_names[best_n]
    acc_top_k = np.mean(np.array([1 if y_test[n] in y_top_k[n] else 0 for n in range(y_test.shape[0])]))
    return  acc_top_k

def nested_cv(X, y, groups, Classifier, param_grid=None, k=1, grid_search=False,
              groupby=None, sess=None, mean_times=None,
              postprocess=False, feature_selection=None):
    """
    Nested Cross validation with fMRI fold

    Parameters
    ----------
    X : array-like of shape(n_samples, n_feautre)
        Training vectors
    y : array-like of shape(n_samples,)
        Target values(class label in classification)
    groups : ndarray
        Groups to constrain the cross-validation. We usually use run_idx in fMRI fold.
    Classifier : sklearn classifier object
        Sklearn classifier.
    param_grid : dict
        Parameters info in corresponding classifier.
    k : int
        Top k acc. Different k will have different chance level.
    grid_search : bool
        if True, the cv will start grid searching based on param_grid
    groupby : str, optional
        Define the cross validtion in which groups.
        The choices are group_run, single_sess and group_sess
        In group_run, the test set will be 4 runs randomly selected from different session
        In single_sess, the train set and test set are using one session data. Note to assign the 
            the sess value if using the single_ses group_by
        In group_sess, the test set will be 1 session randomly selected
    sess : int, optional
        Define the session number when groupby is single_sess. The default is None.
    mean_times : int
        Define the mean times of test set sample in a same class.
    postprocess : bool
        if True, it will generate classification report and confusion_matrix.
        Make sure to adjust the path in function plot_confusion_matrix and save_classification_report
    feature_selection : int 
        The percentage of feature selection in active-based voxel selction.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    outer_scores_single : list
        The score in single trial.
    outer_scores_mean : list
        The score in mean pattern.
    best_params : list
        Best params in grid searching.

    """
    # define containers
    outer_scores_mean = []
    outer_scores_single = []
    best_params = []

    # change groups 
    # group by sess fold, which contains 4 runs from different sess
    if groupby == 'group_run':
        # define cv num
        n_inner_cv, n_outer_cv = 9, 10
        groups_new = np.zeros(groups.shape)
        sess_fold = np.arange(np.unique(groups).shape[0]).reshape(4, 10)
        # shuffle sess
        for idx in range(sess_fold.shape[0]):
            np.random.shuffle(sess_fold[idx])
        # generate new fold
        for idx in range(sess_fold.shape[1]):
            target_runs = sess_fold[:, idx]
            fold_loc = np.asarray([True if x in target_runs else False for x in groups])
            groups_new[fold_loc] = idx
    # group by session, each sess is a fold        
    elif groupby == 'group_sess':
        n_inner_cv, n_outer_cv = 3, 4
        groups_new = np.asarray([x // 10 % 10 for x in groups]) # get session num of each run
    # group by single session, each run is a fold        
    elif groupby == 'single_sess':
        if sess == None:
            raise ValueError('Please assign sess if groupby is single_sess!')
        # define cv num
        n_inner_cv, n_outer_cv = 9, 10
        sess_run = np.arange(10) + (sess-1)*10
        sess_loc = np.asarray([True if x in sess_run else False for x in groups])
        X = X[sess_loc, :]
        y = y[sess_loc]
        groups_new = groups[sess_loc]
        print(X.shape)
        print(f'Nested CV on Sess{sess}')
        
    # select voxels highly responsive if desired
    if feature_selection:
        n_voxel_select = int(X.shape[1]*(feature_selection/100))
        voxel_pattern = np.max(X, axis=0)
        select_loc = np.argsort(-voxel_pattern)[:n_voxel_select]
        X = X[:, select_loc]
       
    # start cross validation
    split_index = 1
    class_name = np.unique(y)
    confusion = np.zeros((10, class_name.shape[0], class_name.shape[0]))
    # handle situation for not group cv
    if groupby == None: 
        model = Classifier
        outer_scores_single = cross_val_score(model, X, y, cv=10)
    else:
        # define groupcv
        inner_cv = GroupKFold(n_splits = n_inner_cv)
        outer_cv = GroupKFold(n_splits = n_outer_cv)
        # group cv
        for train_index, test_index in outer_cv.split(X, y, groups=groups_new):
            # split train test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            groups_cv = groups_new[train_index]
            # in group_run and group_sess, the groups have the same shape but its' content is different
            # as for single sess, the test index should correspond to groups_new
            if groupby == 'single_sess':
                run_test = groups_new[test_index]
            else:
                run_test = groups[test_index]
            # transform mean pattern 
            X_test_mean = np.zeros((1, X.shape[1]))
            y_test_mean = []
            for idx in np.unique(run_test):
                tmp_X = X_test[run_test==idx, :]
                tmp_y = y_test[run_test==idx]
                # loop to tranform mean pattern in each class
                for class_idx in np.unique(y):
                    if mean_times == None:
                        # generate mean pattern for class in each run
                        if class_idx in tmp_y:
                            class_loc = tmp_y == class_idx
                            pattern = np.mean(tmp_X[class_loc], axis=0)[np.newaxis, :]
                            X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                            y_test_mean.append(class_idx)
                    else:
                        # generate mean pattern in specified times
                        animacy_loc = tmp_y == class_idx
                        animacy_X = tmp_X[animacy_loc, :]
                        for mean_idx in range(int(animacy_X.shape[0]/mean_times)):
                            pattern = np.mean(animacy_X[mean_idx*mean_times:(mean_idx+1)*mean_times-1], 
                                              axis=0)[np.newaxis, :]
                            X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                            y_test_mean.append(class_idx)
            
            X_test_mean = np.delete(X_test_mean, 0, axis=0)
            y_test_mean = np.array(y_test_mean)
            # fit grid in inner loop
            if grid_search:
                model = GridSearchCV(Classifier, param_grid, cv=inner_cv, n_jobs=8, verbose=10)
                model.fit(X_train, y_train, groups=groups_cv)
                best_params.append(model.best_params_)
            else:
                model = Classifier
                model.fit(X_train, y_train)
            # handle specified situation on svm
            if param_grid['classifier'][0].__class__.__name__ in ['SVC', 'Lasso'] or groupby == 'single_sess':
                outer_scores_single.append(model.score(X_test, y_test))
                outer_scores_mean.append(model.score(X_test_mean, y_test_mean))
            else:
                # get topk score
                X_probs_mean = model.predict_proba(X_test_mean)
                X_probs = model.predict_proba(X_test)
                # test score in outer loop
                outer_scores_single.append(top_k_acc(X_probs, y_test, k))
                outer_scores_mean.append(top_k_acc(X_probs_mean, y_test_mean, k))
        
            if postprocess:
                # postprocess: including confusion matrix, classification report
                # predict
                y_pred_mean = model.predict(X_test_mean)
                y_pred = model.predict(X_test)
                # plot and save info
                confusion[split_index-1] = confusion_matrix(y_test_mean, y_pred_mean, normalize='true')
                save_classification_report(y_test_mean, y_pred_mean, f'mean_split{split_index}')
                save_classification_report(y_test, y_pred, f'single_split{split_index}')
                
            print(f'Finish cv in split{split_index}')
            split_index += 1
            
        if postprocess:
            confusion = np.mean(confusion, axis=0)
            plot_confusion_matrix(confusion, class_name, 'mean_pattern')

    return outer_scores_single, outer_scores_mean, best_params
        

def class_sample(data, label, run_idx):
    """
    Make each class has the same sample based on the distance 
    between sample and class mean pattern

    Parameters
    ----------
    data : ndarray
        n_sample x n_feature
    label : ndarray
        DESCRIPTION.
    run_idx : ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # sample class
    n_sample = pd.DataFrame(label).value_counts().min()
    data_sample = np.zeros((1, data.shape[1]))
    run_idx_sample = np.zeros((1))
    # loop to sample
    for idx,class_idx in enumerate(np.unique(label)):
        class_loc = label == class_idx 
        class_data = data[class_loc]
        class_mean = np.mean(class_data, axis=0)[np.newaxis, :]
        eucl_distance = distance_matrix(class_data, class_mean).squeeze()
        # random select sample to make each class has the same number
        select_idx = np.argsort(eucl_distance)[:n_sample]
        data_class = data[class_loc, :][select_idx]
        run_idx_class = run_idx[class_loc][select_idx]
        # concatenate on the original array
        data_sample = np.concatenate((data_sample, data_class), axis=0)
        run_idx_sample = np.concatenate((run_idx_sample, run_idx_class), axis=0)
    # prepare final data
    data_sample = np.delete(data_sample, 0, axis=0)
    run_idx_sample = np.delete(run_idx_sample, 0, axis=0)
    label_sample = np.repeat(np.unique(label), n_sample)
    
    return data_sample, label_sample, run_idx_sample
    
    

def find_outlier(data, label, cont):
    # input: data,  contamination -> outlier ratio
    # output: scatter figure;
    # return: X_pca -> PCA processed data, array of float64
    #         y_pred -> outlier mark, array of int64 (1 & -1)
    
    out_index = []
    
    for class_idx in np.unique(label):
        
        # get class data
        class_label = label == class_idx
        class_data = data[class_label, :]
        class_loc = np.where(class_label==1)[0]
        
        # scaler
        scaler = StandardScaler()
        scaler.fit(class_data)
        X_scaled = scaler.transform(class_data)
        
        # PCA
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        
        # EllipticEnvelope to find outlier
        esti = EllipticEnvelope(contamination=cont)
        y_pred = esti.fit(X_pca).predict(X_pca)
       
        # store outlier index
        out_index.extend(class_loc[np.where(y_pred == -1)[0]])
        print(f'Finish finding outlier index in class {class_idx}')

    # return
    return out_index


def gen_param_grid(method):
    
    param_grid = {
                  'svm':    
                   {'classifier': [SVC(max_iter=8000)], 'feature_selection':[SelectPercentile()],
                    'classifier__C': [0.001],
                    'classifier__kernel': ['linear'],
                    'classifier__decision_function_shape': ['ovo'],
                    'feature_selection__percentile': [25],},
                  'logistic':    
                      {'classifier': [LogisticRegression(max_iter=8000)], 
                       'feature_selection':[SelectPercentile()],
                       'classifier__C': [0.001],
                       'classifier__solver': ['liblinear'],
                       'feature_selection__percentile': [25],},
                  'rf':    
                      {'classifier': [RandomForestClassifier()], 'feature_selection':[SelectPercentile()],
                       'classifier__n_estimators': [500, 300, 200, ],
                       'feature_selection__percentile': [25],},
                  'mlp':
                      {'classifier': [MLPClassifier()], 'feature_selection':[SelectPercentile()],
                       'classifier__alpha': [0.01],
                       'classifier__hidden_layer_sizes': [(200,)],
                       'feature_selection__percentile': [25],},
                  'lasso':    
                      {'classifier': [Lasso(max_iter=8000)],
                       'classifier__alpha': [0.001, 0.01, 0.1, 1],}, 
                  'lda':    
                   {'classifier': [LinearDiscriminantAnalysis()], 'feature_selection':[SelectPercentile()],
                    'classifier__solver': ['lsqr'],
                    'classifier__shrinkage': [0.9],
                    'feature_selection__percentile': [25],},
                  }  

    return param_grid[method]            


