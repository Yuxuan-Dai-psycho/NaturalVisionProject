import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile

def nested_cv(X, y, groups, Classifier, param_grid, groupby='group_run', sess=None):
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
        
    # define groupcv
    inner_cv = GroupKFold(n_splits = n_inner_cv)
    outer_cv = GroupKFold(n_splits = n_outer_cv)
    # define containers
    outer_scores_mean = []
    outer_scores_single = []
    best_params = []

    # start cross validation
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
            for class_idx in np.unique(y):
                if class_idx in tmp_y:
                    class_loc = tmp_y == class_idx
                    pattern = np.mean(tmp_X[class_loc], axis=0)[np.newaxis, :]
                    X_test_mean = np.concatenate((X_test_mean, pattern), axis=0)
                    y_test_mean.append(class_idx)
        X_test_mean = np.delete(X_test_mean, 0, axis=0)
        y_test_mean = np.array(y_test_mean)
        # fit grid in inner loop
        grid = GridSearchCV(Classifier, param_grid, cv=inner_cv, n_jobs=8, verbose=10)
        grid.fit(X_train, y_train, groups=groups_cv)
        # test score in outer loop
        outer_scores_single.append(grid.score(X_test, y_test))
        outer_scores_mean.append(grid.score(X_test_mean, y_test_mean))
        best_params.append(grid.best_params_)
        
    return outer_scores_single, outer_scores_mean, best_params
        



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
                    'classifier__C': [0.001, 0.01],
                    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'feature_selection__percentile': [30],},
                  'logistic':    
                      {'classifier': [LogisticRegression(max_iter=8000)], 
                       'feature_selection':[SelectPercentile()],
                       'classifier__C': [0.001],
                       'classifier__solver': ['newton-cg', 'liblinear'],
                       'feature_selection__percentile': [30]},
                  'random_forest':    
                      {'classifier': [RandomForestClassifier()], 'feature_selection':[SelectPercentile()],
                       'classifier__C': [0.001, 0.01],
                       'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                       'feature_selection__percentile': [30],},
                  }  

    return param_grid[method]            



















