import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.metrics.pairwise import pairwise_kernels


def getQuery(ref_query_metric, i, N):
    row_metric = list(ref_query_metric[i])
    maxN = sorted(range(len(row_metric)), key=lambda sub: row_metric[sub])[-N:]
    return maxN

def getQuery_parallel(ref_query_metric, i, N):
    res = [getQuery(ref_query_metric, j, N) for j in i]
    return res

def single_query(ref_query_metric, N1):
    N_rows, _ = ref_query_metric.shape  # 细胞基因
    return [getQuery(ref_query_metric, i, N1) for i in range(N_rows)]

def query_helper(args):
    return args[0],getQuery_parallel(*args[1])

def parallel_query(ref_query_metric, N, n_jobs=32):
    n_jobs =64
    ref_query_metric = np.array(ref_query_metric) 
    p = mp.Pool(n_jobs)
    job_args = [(j,(ref_query_metric[i], range(len(i)), N)) for j,i in enumerate(np.array_split(range(ref_query_metric.shape[0]), n_jobs))]

    lst = []
    for args in job_args:
        p.apply_async(query_helper, (args,), callback=lst.append)

    p.close()
    p.join()

    lst = sorted(lst, key=lambda x: x[0])
    lst = [i[1] for i in lst]
    res = []
    for i in lst:
        res.extend(i)

    return res

def find_mutual_nn(ref_query_metric, N1=3, N2=3, n_jobs=1):
    N_rows, N_cols = ref_query_metric.shape
    time0 = time.time()
    if n_jobs == 1:
        # 获取前N1个度量的坐标
        k_index_1 = single_query(ref_query_metric.T, N2) # 2->1 数据2的前N1个度量的1的index
        k_index_2 = single_query(ref_query_metric, N1) # 1->2 数据1的前N2个度量的2的index
    else:
        k_index_1 = parallel_query(ref_query_metric.T, N2, n_jobs=n_jobs)
        k_index_2 = parallel_query(ref_query_metric, N1,n_jobs=n_jobs)
    time1 = time.time()
    print('knn time is {:.2f} s'.format(time1 - time0))
    mutual_1 = []
    mutual_2 = []
    for index_2 in range(N_cols):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
    time2 = time.time()
    print('mnn time is {:.2f} s'.format(time2 - time1))
    return mutual_1, mutual_2


def filterPairs(arr, ref_query_metric, N1=2561, N2=2561, n_jobs=1):
    N_rows, N_cols = ref_query_metric.shape

    if n_jobs < 2:
        k_index_1 = single_query(ref_query_metric.T, N2) # 2->1 数据2的前N1个度量的1的index
        k_index_2 = single_query(ref_query_metric, N1) # 1->2 数据1的前N2个度量的2的index
    else:
        k_index_1 = parallel_query(ref_query_metric.T, N2, n_jobs=n_jobs)
        k_index_2 = parallel_query(ref_query_metric, N1,n_jobs=n_jobs)
    arr1 = np.array([0, 0])
    for i in range(arr.shape[0]):
        if (arr.iloc[i, 0] in k_index_1[arr.iloc[i, 1]]) and (arr.iloc[i, 1] in k_index_2[arr.iloc[i, 0]]):
            arr1 = np.vstack((arr1, arr.iloc[i, :]))
    arr1 = np.delete(arr1, (0), axis=0)

    return pd.DataFrame(arr1)


def selectPairs(df, similarity_matrix, N=3):
    weight_df = pd.DataFrame([[row[0], row[1],
                               similarity_matrix.iloc[row[0], row[1]]]
                              for index, row in df.iterrows()])

    g = []
    for i in range(2):
        g.append(weight_df.
                 groupby([i]).
                 apply(lambda x: x.sort_values([2], ascending=True)).
                 reset_index(drop=True).groupby([i]).head(N))

    g1 = pd.concat(g, ignore_index=True).drop_duplicates()

    return pd.concat(g, ignore_index=True).iloc[:, [0, 1]].drop_duplicates(), g1


def mnn_from_counts(count1, count2, N1, N2, N, n_jobs):
    # get similiarity matrix
    start = time.time()
    similarity_selected = pd.DataFrame(
        pairwise_kernels(count1,
                         count2,
                         metric='cosine')
    )
    ref_pair, query_pair = find_mutual_nn(similarity_selected,
                                          N1=N1,
                                          N2=N2,
                                          n_jobs=n_jobs)
    pair_ref_query = pd.DataFrame([ref_pair, query_pair]).T
    pair_ref_query.drop_duplicates()
    pair_ref_query, g1 = selectPairs(pair_ref_query, similarity_selected,
                                      N=N)
    pair_ref_query = pd.DataFrame(pair_ref_query)
    end = time.time()
    print('the time of compute mnn is {:.2f} s'.format(end - start))
    return pair_ref_query, g1