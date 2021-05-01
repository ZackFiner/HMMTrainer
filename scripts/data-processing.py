import ctypes
import numpy as np
import pandas as pd
from sklearn import metrics

hmm_dll = ctypes.WinDLL("K:\\GitHub\\CS185C-HMM\\x64\\dll\\CS185C-HMM.dll")

zero_access = ("zeroaccess", "za", 1305)
winwebsec = ("winwebsec", "ww", 4360)
zbot = ("zbot", "zb", 2136)

# Taken From https://dbader.org/blog/python-ctypes-tutorial-part-2
def wrap_func(lib, func_name, return_type, arg_types):
    func = lib.__getattr__(func_name)
    func.restype = return_type
    func.argtypes = arg_types
    return func
    

retrieve_rocData = wrap_func(hmm_dll, 'getRoc', None, [ctypes.POINTER(ctypes.c_float), 
                                                           ctypes.c_uint, 
                                                           ctypes.c_char_p, 
                                                           ctypes.c_char_p, 
                                                           ctypes.c_char_p, 
                                                           ctypes.c_uint])

retrieve_rocData_fold = wrap_func(hmm_dll, 'scoreModelFolds', None, [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_uint,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint])

evaluate_hmm_folds = wrap_func(hmm_dll, 'evaluateModelFolds', None, [
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_uint,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint])

def get_roc_fold_data(buffer_size, hmm, pos, neg, clip):
    bhmm = hmm.encode('utf-8')
    bpos = pos.encode('utf-8')
    bneg = neg.encode('utf-8')
    results_buffer = (ctypes.c_float * buffer_size)()
    
    retrieve_rocData(results_buffer, buffer_size, bpos, bneg, bhmm, clip)
    
    arr = np.array(results_buffer)
    arr = np.reshape(arr, (int(buffer_size/2), 2))
    return arr





current_target = ()
def get_fold_data(host_hmm, negative_family, series, n, m, eval_size = 0):
    folds = list()
    buffer_size = (host_hmm[2] + negative_family[2])*2*10
    results_buffer = (ctypes.c_float * buffer_size)()
    
    c_dset1 = "K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\{family}".format(family=host_hmm[0]).encode('utf-8')
    c_dset2 = "K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\{family}".format(family=negative_family[0]).encode('utf-8')
    c_hmm = "K:\\GitHub\\CS185C-HMM\\hmms\\{family}".format(family=host_hmm[0]).encode('utf-8')
    c_series = series.encode('utf-8')
    c_abrv = host_hmm[1].encode('utf-8')
    retrieve_rocData_fold(results_buffer, buffer_size, 
                        c_dset1,
                        c_dset2,
                        c_hmm,
                        c_series,
                        c_abrv,
                        m,
                        n,
                        eval_size)

    folds = np.array(results_buffer)
    folds = folds.reshape(10, host_hmm[2] + negative_family[2], 2)

    return folds

def evaluate_model(host_hmm, negative_family, series, n, m, eval_size=0):
    buffer_size = (host_hmm[2] + negative_family[2])*10
    results_buffer = (ctypes.c_float * buffer_size)()
    
    c_dset1 = "K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\{family}".format(family=host_hmm[0]).encode('utf-8')
    c_dset2 = "K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\{family}".format(family=negative_family[0]).encode('utf-8')
    c_hmm = "K:\\GitHub\\CS185C-HMM\\hmms\\{family}".format(family=host_hmm[0]).encode('utf-8')
    c_series = series.encode('utf-8')
    c_abrv = host_hmm[1].encode('utf-8')
    
    evaluate_hmm_folds(results_buffer, buffer_size,
                       c_dset1,
                       c_dset2,
                       c_hmm,
                       c_series,
                       c_abrv,
                       m,
                       n,
                       eval_size)
    
    folds = np.array(results_buffer)
    folds = folds.reshape(10, host_hmm[2] + negative_family[2])
    
    return folds

def evaluate_model_folds(host_hmm, negative_family, series, n, m, eval_size=0):
    folds = evaluate_model(host_hmm, negative_family, series, n, m, eval_size)
    
    fold_list = list()
    entries = 0
    
    fold_size = int(host_hmm[2] / 10)
    for i in range(9):
        test_size = fold_size + negative_family[2]
        labels = np.zeros(test_size)
        for j in range(fold_size):
            labels[j] = 1
        scores = np.concatenate((folds[i, i*fold_size : (i+1)*fold_size], folds[i, host_hmm[2]:]))
        fold_list.append(np.array([labels, scores]))
        entries = entries + fold_size
    
    pos_size = host_hmm[2] - entries
    test_size = pos_size + negative_family[2]
    labels = np.zeros(test_size)
    for j in range(pos_size):
        labels[j] = 1
    scores = np.concatenate((folds[i, 9*fold_size : host_hmm[2]], folds[i, host_hmm[2]:]))
    fold_list.append(np.array([labels, scores]))

    return fold_list

def get_data(host_hmm, hmm_fpath, negative_family):
    buffer_size = (host_hmm[2] + negative_family[2])*2
    pos = "K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\{family}".format(family = host_hmm[0])
    neg = "K:\\GitHub\\CS185C-HMM\\Debug\\training_data\\{family}".format(family = negative_family[0])
    data = get_roc_fold_data(buffer_size, hmm_fpath, pos, neg, 0)
    return data

def plot_fold_rocs(fold_data, hmm, negatives, n, m):
    fig, roc = plt.subplots(1)
    fig.suptitle("{hmm} vs. {negatives} (N={n}, m={m})".format(hmm=hmm[0], negatives=negatives[0], n=n, m=m))
    roc.set_title("ROC")
    roc.set_xlabel('FPR')
    roc.set_ylabel('TPR')
    for i in range(10):
        fpr, tpr, thresholds = metrics.roc_curve(fold_data[i][0], fold_data[i][1], pos_label=1)
        roc.plot(fpr, tpr, label='fold-{index}'.format(index=i))
    roc.plot([0,1], [0,1], 'r--', label='Random')
    roc.legend()
    return fig, roc

def get_ideal_accuracy(pos_set, neg_set):
    pos_set = np.sort(pos_set)
    pos_set_l = pos_set.shape[0]
    neg_set = np.sort(neg_set)
    neg_set_l = neg_set.shape[0]
    
    total_size_div = 1/(len(pos_set) + len(neg_set))
    best_accuracy = 0
    best_thresh = 0
    best_sens = 0
    best_spec = 0
    min_dist_to_best = 500
    
    j = 0
    for i in range(pos_set_l): # for all positive threshholds
        threshold = pos_set[i]
        TP = pos_set_l - i # since we're using the threshold that is <= pos_set[i], 
                           # then all points greater than or equal to this are true positives
        while(neg_set[j] < threshold and j < (neg_set_l-1)):
            j+=1
        TN = max(j-1, 0) ## J is all values less than the threshold, therefore this is the TN for the given threshold
        temp_acc = (TP + TN) * total_size_div
        temp_sens = TP / pos_set_l
        temp_spec = TN / neg_set_l
        TPR = temp_sens
        FPR = 1 - temp_spec
        score =  (1-TPR)**2 + FPR**2
        if min_dist_to_best > score:
            best_accuracy = temp_acc
            best_thresh = threshold
            best_sens = temp_sens
            best_spec = temp_spec
            min_dist_to_best = score
            
    j = 0
    for i in range(neg_set_l): # for all positive threshholds
        threshold = neg_set[i]
        TN = max(i-1, 0) # is all values less than or equal to threshold
        while(pos_set[j] < threshold and j < (pos_set_l-1)):
            j+=1
        TP = pos_set_l - j ## J is all values less than the threshold, therefore the TP is pos_size - j
        temp_acc = (TP + TN) * total_size_div
        temp_sens = TP / pos_set_l
        temp_spec = TN / neg_set_l
        TPR = temp_sens
        FPR = 1 - temp_spec
        score =  (1-TPR)**2 + FPR**2
        if min_dist_to_best > score:
            best_accuracy = temp_acc
            best_thresh = threshold
            best_sens = temp_sens
            best_spec = temp_spec
            min_dist_to_best = score

            
    return best_accuracy, best_thresh, best_spec, best_sens
        
    

def get_data_analysis(fold_data, hmm, negatives, n, m):
    buff = np.zeros((10, 5))
    for i in range(10):
        auc = metrics.roc_auc_score(fold_data[i][0], fold_data[i][1])
        pos_count = int(np.sum(fold_data[i][0]))
        pos_set = fold_data[i][1, 0:pos_count]
        neg_set = fold_data[i][1, pos_count:]
        
        ideal_accuracy, thresh, spec, sens = get_ideal_accuracy(pos_set, neg_set)
        
        buff[i, :] = [auc, ideal_accuracy, thresh, spec, sens]
    dframe = pd.DataFrame(buff, columns = ["AUC", "Ideal Accuracy (*)", "Ideal Accuracy Thresh.", "Specificity", "Sensitivity"])
    dframe.loc["mean"] = dframe.mean()
    dframe.loc["max"] = dframe.max()
    return dframe


def plot_scatter(fold_data, fold_index, hmm, negatives, n, m):
    fig, scatter = plt.subplots(1)
    fig.suptitle("{hmm} vs. {negatives} (N={n}, m={m})".format(hmm=hmm[0], negatives=negatives[0], n=n, m=m))
    scatter.set_title("Scatter Plot")
    scatter.set_ylabel('LogProb(O|model)/|O|')
    fold = fold_data[fold_index]
    pos_size = int(np.sum(fold[0]))
    pos_x = np.random.rand(pos_size)*3.0
    neg_x = np.random.rand(len(fold[0])-pos_size)*3.0
    
    scatter.plot(neg_x, fold[1, pos_size:], '^', color="blue", label='{neg} O'.format(neg=negatives[0]))
    scatter.plot(pos_x, fold[1, :pos_size], 'o', color="red", label='{hmm} O'.format(hmm=hmm[0]))
    scatter.legend()
    return fig, scatter