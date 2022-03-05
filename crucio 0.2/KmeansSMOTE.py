'''
Created with love by Sigmoid
​
@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import random
import sys
from sklearn.cluster import KMeans
from random import randrange
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from .erorrs import NotBinaryData, NoSuchColumn, NoSuchMethodError, MissingDataError, DifferentColumnsError

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

class KmeansSMOTE:


    def __init__(self, k: "int > 0" = 5, seed: float = 42, irt = 0.5, n_synth = 100, n_clusters = 2, de : int = None,
                 binary_columns : list = None, init : ['k-means++', 'random'] = 'k-means++', n_init : int = 10, max_iter : int = 300,
                tol : float = 1e-4, random_state : int = None, copy_x : bool = True, algorithm : ['auto', 'full', 'elkan'] = 'auto') -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        :param seed: int, default = 42
            seed for random
        :param irt: float, default = 0.5
            imbalance ratio threshold
        :param n_synth: int, default = 10
            Number of samples to be generated
        param n_clusters: int, n_clusters > 0, default = 2
            Number of clusters that the algorithm will search for
        param de: int, default = None
            exponent used for computing of density, default to the number of features in data frame
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param init: {‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’
            Kmeans Method for initialization
        :param n_init: int, default=10
            Number of time the k-means algorithm will be run with different centroid seeds. 
        :param max_iter int, default=300
            Maximum number of iterations of the k-means algorithm for a single run.
        ** :param tol float, default=1e-4
            Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
        :param random_state int, RandomState instance or None, default=None
            Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        :param copy_x bool, default=True
            When pre-computing distances it is more numerically accurate to center the data first. 
            If copy_x is True (default), then the original data is not modified. 
            If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean. 
        :param algorithm: {“auto”, “full”, “elkan”}, default=”auto”
            K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation is more efficient on data with well-defined clusters, by using the triangle inequality.         
        '''

        self.__k = k
        self.irt = irt
        self.n_clusters = n_clusters
        self.n_synth = n_synth
        self.de = de
        if binary_columns is None:
            self.__binarize = False
            self.__binary_columns = None
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns

        self.kmeans = KMeans(n_clusters=self.n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                            random_state=random_state, copy_x=copy_x, algorithm=algorithm)
        self.__seed = seed
        np.random.seed(self.__seed)
        random.seed(self.__seed)

    def __to_binary(self) -> None:
        '''
            If the :param binary_columns: is set to True then the intermediate values in binary columns will be rounded.
        '''

        for column_name in self.__binary_columns:
            serie = self.synthetic_df[column_name].values
            threshold = (self.df[column_name].max() + self.df[column_name].min()) / 2
            for i in range(len(serie)):
                if serie[i] >= threshold:
                    serie[i] = self.df[column_name].max()
                else:
                    serie[i] = self.df[column_name].min()
            self.synthetic_df[column_name] = serie

    def __infinity_check(self, matrix : 'np.array') -> 'np.array':
        '''
            This function replaces the infinity and -infinity values with the minimal and maximal float python values.
        :param matrix: 'np.array'
            The numpy array that was generated my the algorithm.
        :return: 'np.array'
            The numpy array with the infinity replaced values.
        '''

        matrix[matrix == -np.inf] = sys.float_info.min
        matrix[matrix == np.inf] = sys.float_info.max
        return matrix

    def balance(self, df : pd.DataFrame, target : str) -> pd.DataFrame:
        '''
            Balance all minority classes to the number of majority class instances
        :param df: pandas DataFrame
             Data Frame on which the algorithm is applied
        :param y_column: string
             The target name of the value that we have to predict
        '''

        if target not in df.columns:
            raise NoSuchColumn(f"{target} isn't a column of passed data frame")
            
        if self.__binary_columns:
            for column in self.__binary_columns:
                if column not in df.columns:
                    raise DifferentColumnsError(f"The passed data frame doesn't contain the {column} column passed to the binary columns.")
        
        if df.isna().any().any():
            raise MissingDataError("The passed data frame contains missing values!") 
        
        if not df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
            raise NonNumericDataError("The given DataFrame contains non-numeric values !")
        
        if self.de is None:
            self.de = len(df.columns)-1
        
        #get unique values of target column
        unique = df[target].unique()

        self.df = df.copy()

        #training columns
        self.X_columns = [column for column in self.df.columns if column != target]
        self.synthetic_data = []
        self.synthetic_final_df = self.df.copy()
        classes_nr_samples = []

        #Find the majority class and creating nr of minority calss samples list
        for cl in unique:
            classes_nr_samples.append(len(df[df[target]==cl]))
            majority_class_nr_samples = max(classes_nr_samples)
        majority_class = unique[np.argmax(classes_nr_samples)]

        #set arrays witch indicates majority and minority classes
        minority_classes = [cl for cl in unique if cl!=majority_class]
        minority_classes_samples = [cl for i,cl in enumerate(classes_nr_samples) if i!=np.argmax(classes_nr_samples)]

                
        filtered_clusters = []

        X = self.df.drop(target,1).values
        self.kmeans.fit(X) 
        X = self.df.copy()
        X['cluster'] = self.kmeans.labels_
        
        for cluster in range(self.n_clusters):
            cluster_data = X.loc[X['cluster']==cluster,:]

            imbalance_ratio = (len(cluster_data.loc[cluster_data[target]!=majority_class])+1)/(len(cluster_data.loc[cluster_data[target]==majority_class])+1)
            if imbalance_ratio >= self.irt:
                filtered_clusters.append(cluster)
        sparsity_sum = 0

        sparsity_factors = []
        for cluster in filtered_clusters:
            # compute sampling weight based on its minority density
            minority_cluster_data = X.loc[(X['cluster']==cluster) & (X[target]!=majority_class), self.X_columns] 
            # get average euclidian distance between every pair of data in minority_cluster_data and calculate its mean

            distances_df = pd.DataFrame(cdist(minority_cluster_data.values, minority_cluster_data.values))
            average_minority_distance = distances_df.to_numpy().sum()/(len(distances_df)*(len(distances_df)-1))


            density_factor = len(minority_cluster_data)/average_minority_distance
            sparsity_factor = 1/density_factor
            sparsity_factors.append(sparsity_factor)

            sparsity_sum+=sparsity_factor
            
        sampling_weight = [x / sparsity_sum for x in sparsity_factors] 

        for i, cluster in enumerate(filtered_clusters):

            cluster_data = X.loc[(X['cluster']==cluster), self.X_columns + [target]]
            # nr of samples to generate should give lol 
            lenght = len(cluster_data)
            
            generated_samples  = SMOTE(self.__k, self.__seed, self.__binary_columns).balance(cluster_data, target)
            self.synthetic_df = generated_samples.head(len(generated_samples)-lenght)
            
            #to all dataset add new rows
            self.synthetic_final_df = pd.concat([self.synthetic_df,self.synthetic_final_df],axis=0)
            
        self.synthetic_df = self.synthetic_final_df.copy()

        #return new df
        return self.synthetic_df