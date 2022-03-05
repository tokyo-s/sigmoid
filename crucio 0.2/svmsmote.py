'''
Created with love by Sigmoid
​
@Author - Stojoc Vladimir - vladimir.stojoc@gmail.com
'''
import numpy as np
import pandas as pd
import random
import sys
from random import randrange
from sklearn.svm import SVC
from .erorrs import NotBinaryData, NoSuchColumn, NoSuchMethodError, MissingDataError, DifferentColumnsError

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

class SVMSMOTE:


    def __init__(self,k: "int > 0" = 5, m: int = 10, percent : int = 100, seed: float = 42, binary_columns : list = None,
                C=1.0, kernel='rbf', degree = 3, gamma = 'scale', coef0 = 0.0, shrinking = True, probability = False, tol=1e-3,
                 cache_size=200, class_weight=None, max_iter = -1, decision_function_shape='ovr', break_ties=False, random_state=None) -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        :param m: int m>0, default = 10
            Number
        :param percent: int, percent>0, default = 100
            Percent with witch the length of the data set should increase
        :param seed: int, default = 42
            seed for random
        :param binary_columns: list, default = None
            The list of columns that should have binary values after balancing.
        :param C: float, default = 1.0
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. 
        :param kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
            Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used.
        :param degree: int, default = 3
            Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
        :param gamma: {‘scale’, ‘auto’} or float, default=’scale’
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        :param coef0: float, default = 0.0
            Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
        :param shrinking: bool, default = True
            Whether to use the shrinking heuristic.
        :param probability: bool, default = False
            Whether to enable probability estimates. This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict.
        :param tol: float, default = 1e-3
            Tolerance for stopping criterion.
        :param cache_size: float, default = 200
            Specify the size of the kernel cache (in MB).
        :param class_weight: dict or ‘balanced’, default = None
            Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        :param max_iter: int, default = -1
            Hard limit on iterations within solver, or -1 for no limit.    
        :param decision_function_shape: {‘ovo’, ‘ovr’}, default = ’ovr’
            Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). 
        :param break_ties: bool, default = False
            If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned.
        :param random_state: int, RandomState instance or None, default = None
            Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when probability is False. Pass an int for reproducible output across multiple function calls. 
        '''

        self.__k = k
        self.__m = m
        if binary_columns is None:
            self.__binarize = False
            self.__binary_columns = []
        else:
            self.__binarize = True
            self.__binary_columns = binary_columns

        self.percent = percent
        
        self.svm = SVC(C=C, kernel=kernel, degree = degree, gamma = gamma, coef0 = coef0, shrinking = shrinking,
                       probability = probability, tol=tol, cache_size=cache_size, class_weight=class_weight, max_iter = max_iter,
                       decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state) 

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
        
        #get unique values of target column
        unique = df[target].unique()


        self.df = df.copy()

        #training columns
        self.X_columns = [column for column in self.df.columns if column != target]
        self.synthetic_data = []
        self.synthetic_final_df = self.df.copy()
        classes_nr_samples = []

        # TODO: also should think if there are more majority classes, should divide by number of minority classes

        #getting total number of samples that will be generated
        T = self.percent/100 * len(self.df)

        #Find the majority class and creating nr of minority calss samples list
        for cl in unique:
            classes_nr_samples.append(len(df[df[target]==cl]))
        majority_class = unique[np.argmax(classes_nr_samples)]

        #set arrays witch indicates majority and minority classes
        minority_classes = [cl for cl in unique if cl!=majority_class]

        self.svm.fit(self.df[self.X_columns], self.df[target])
        support = self.df.iloc[self.svm.support_, :]
        

        #Smote algorithm, oversamples for every minority class
        for i,minority_class in enumerate(minority_classes):

            # calculate the amount, nr of sampled examples for every minority class, evenly distributed from T
            nr_of_synthetic_samples = int(round(T/len(support.loc[support[target]==minority_class])))
            if nr_of_synthetic_samples==0:
                nr_of_synthetic_samples = 1
            
            minority_class_support_vectors = support.loc[support[target]==minority_class, self.X_columns].values

            for minority_support_vector in minority_class_support_vectors:
                
                #get k neighbours
                nn_indexes = self.__get_k_neighbours(minority_support_vector,self.df.loc[self.df[target]==minority_class,self.X_columns].values, self.__k)
                nn = self.df.iloc[nn_indexes, :]
                nn = nn.loc[:, self.X_columns].values
                m_neighbours_indexes = self.__get_k_neighbours(minority_support_vector,self.df.loc[:, self.X_columns].values, self.__m) # should put m neighbours

                m_neighbours = self.df.iloc[m_neighbours_indexes, :]
                m_majority = len(m_neighbours.loc[m_neighbours.loc[:,target]==majority_class])


                if 0<= m_majority< self.__m/2:
                    for _ in range(nr_of_synthetic_samples):
                        j = random.randint(0,len(nn)-1)
                        p = np.random.rand()
                        new_example = minority_support_vector + p * (minority_support_vector - nn[j])
                        self.synthetic_data.append(new_example)
                elif self.__m/2 <= m_majority < self.__m:
                    for _ in range(nr_of_synthetic_samples):
                        j = random.randint(0,len(nn)-1)
                        p = np.random.rand()
                        new_example = minority_support_vector + p * (nn[j] - minority_support_vector)
                        self.synthetic_data.append(new_example)
                
            self.synthetic_data = self.__infinity_check(np.array(self.synthetic_data))
            self.synthetic_df = pd.DataFrame(np.array(self.synthetic_data), columns=self.X_columns)
            self.synthetic_data = []
            self.synthetic_df.loc[:, target] = minority_class
            
            #to all dataset add new rows
            self.synthetic_final_df = pd.concat([self.synthetic_df,self.synthetic_final_df],axis=0)
        
        self.synthetic_df = self.synthetic_final_df.copy()

        # Rounding binary columns if needed.
        if self.__binarize:
            self.__to_binary()

        #return new df
        return self.synthetic_df
    
    def __get_k_neighbours(self, example : 'np.array', minority_samples : 'np.ndarray', k : 'int > 0') -> 'np.array':
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param minority_samples: Numpy.ndarray
            minority class samples from where we find neighbours
        :param k: int, k>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points
        '''
        distances = []

        #find all distances from example to every sample in minoriry_sample
        for x in minority_samples:
            distances.append(np.linalg.norm(x - example, ord=2))

        #select k neighbours
        predicted_index = np.argsort(distances)[1:k + 1]
        return predicted_index