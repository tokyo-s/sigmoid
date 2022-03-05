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
from .erorrs import NotBinaryData, NoSuchColumn, NoSuchMethodError, MissingDataError

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn


class SPIDER:


    def __init__(self,k: "int > 0" = 3, k2: "int > 0" = 5, seed: float = 42, amplification_type : str = 'weak', relabeling : bool = True) -> None:
        '''
            Setting up the algorithm
        :param k: int, k>0, default = 3
            Number of neighbours which will be considered when looking for simmilar data points
        :param k2: int, k2>0, default = 5
            Number of neighbours which will be considered when looking for simmilar data points, used in 'strong' amplification type
        :param seed: int, default = 42
            seed for random
        :param amplification_type: ['weak', 'strong']
            The type of method to be used
        :param relabeling: bool, default = True
            Whether can relabel majority class samples to minority ones
        '''

        if amplification_type not in ['weak', 'strong']:
            raise NoSuchMethodError(f"{amplification_type} amplification type isn't supported right now, choose one of the following: 'weak' or 'strong'")

        self.__k = k
        self.__k2 = k2
        self.__seed = seed
        np.random.seed(self.__seed)
        random.seed(self.__seed)

        self.amplification_type = amplification_type
        self.relabeling = relabeling


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
        
        if df.isna().any().any():
            raise MissingDataError("The passed data frame contains missing values!") 
        
        if not df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
            raise NonNumericDataError("The given DataFrame contains non-numeric values !") 

            
        #get unique values of target column
        unique = df[target].unique()
        
        self.df = df.copy()
        self.target = target

        #training columns
        self.X_columns = [column for column in self.df.columns if column != target]
        self.X = self.df[self.X_columns]
        self.synthetic_data = []
        self.synthetic_final_df = self.df.copy()
        self.df_copy = self.df.copy()
        self.synthetic_df = pd.DataFrame()
        classes_nr_samples = []
        self.D = np.array([])
        
        #Find the majority class
        for cl in unique:
            classes_nr_samples.append(len(df[df[target]==cl]))
        majority_class = unique[np.argmax(classes_nr_samples)]

        #set arrays witch indicates majority and minority classes
        minority_classes = [cl for cl in unique if cl!=majority_class]
        self.minority_class = minority_classes[0]
        flag = np.zeros(len(self.df)) #1 safe, 0 noisy

        for i, x in enumerate(self.df.loc[:,self.X_columns].values):
            flag[i] = (df.iloc[i][target]==self.__classify_knn(x,self.__k)) 

        self.df['safe'] = pd.Series(flag)
        minority_class_samples = self.df[self.df[target].isin(minority_classes)][self.X_columns + ['safe']]
        
        # Noisy majority class Data Points
        for i in range(len(self.df)):
            if not self.df.iloc[i].safe and self.df.iloc[i][target]==majority_class:
                self.D = np.append(self.D, self.df.index[i]) 
                
        # Weak and no relabeling method
        if self.amplification_type == 'weak' and not self.relabeling:
            for i in range(len(minority_class_samples)):
                if minority_class_samples.iloc[i].safe == 0:
                    to_amplifie = self.__get_k_neighbours(minority_class_samples[self.X_columns].iloc[i].values, self.__k, majority_class, 1)
                    self.synthetic_df = pd.concat([to_amplifie,self.synthetic_df],axis=0)

        # Weak + relabeling method
        elif self.amplification_type == 'weak' and self.relabeling:
            for i in range(len(minority_class_samples)):
                if minority_class_samples.iloc[i].safe == 0:
                    to_amplifie = self.__get_k_neighbours(minority_class_samples[self.X_columns].iloc[i].values, self.__k, majority_class, 1)
                    self.synthetic_df = pd.concat([to_amplifie,self.synthetic_df],axis=0)
            for i in range(len(minority_class_samples)):
                if minority_class_samples.iloc[i].safe == 0:
                    y_neighbors = self.__get_k_neighbours(minority_class_samples[self.X_columns].iloc[i].values, self.__k, majority_class, 0)
                    for y in range(len(y_neighbors)):
                        index = y_neighbors.index[y]
                        self.df_copy.loc[index][target] = self.minority_class
                        self.D = np.delete(self.D, np.where(self.D == index))

        # Strong method
        elif self.amplification_type == 'strong':
            for i in range(len(minority_class_samples)):
                if minority_class_samples.iloc[i].safe == 1:
                    to_amplifie = self.__get_k_neighbours(minority_class_samples[self.X_columns].iloc[i].values, self.__k, majority_class, 1)
                    self.synthetic_df = pd.concat([to_amplifie,self.synthetic_df],axis=0)

            for i in range(len(minority_class_samples)):
                if minority_class_samples.iloc[i].safe == 0:
                    if self.__classify_knn(self.df.loc[minority_class_samples.index[i], self.X_columns].values, self.__k2) == self.df.loc[minority_class_samples.index[i], target]:
                        to_amplifie = self.__get_k_neighbours(minority_class_samples[self.X_columns].iloc[i].values, self.__k, majority_class, 1)
                        self.synthetic_df = pd.concat([to_amplifie,self.synthetic_df],axis=0)
                    else:
                        to_amplifie = self.__get_k_neighbours(minority_class_samples[self.X_columns].iloc[i].values, self.__k2, majority_class, 1)
                        self.synthetic_df = pd.concat([to_amplifie,self.synthetic_df],axis=0)
        
        # Remove from self.df_copy all rows with index that is present in D
        self.df_copy = self.df_copy.drop(self.D)

        #to all dataset add new rows
        self.synthetic_df = pd.concat([self.synthetic_df,self.df_copy],axis=0)

        #return new df
        return self.synthetic_df
    
    def __classify_knn(self, example : 'np.array', k : 'int > 0') -> int or str:
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from data set to get neighbours from
        :param k: int
            Number of neighbours which will be considered when looking for simmilar data points
        '''
        distances = []

        #find all distances from example to every sample in minoriry_sample
        for row in self.X.values:
            distances.append(np.linalg.norm(example - row, ord=2))

        #select k neighbours
        predicted_index = np.argsort(distances)[1:k + 1]
        predicted_labels = self.df.iloc[predicted_index, :][self.target].values

        #return majority class
        return max(set(predicted_labels), key=predicted_labels.tolist().count)

    def __get_k_neighbours(self, example : 'np.array', k : 'int > 0', clas : str or int, flag : bool) -> 'pd.DataFrame':
        '''
            KNN, getting nearest neighbors
        :param example: Numpy array
            the sample row from minority class to get neighbours from
        :param k: int
            Number of neighbours which will be considered when looking for simmilar data points
        :param clas: str or int
            Whitch class samples should be amplified
        :param flag: bool
            Safe or noisy amplification
        '''
        distances = []

        #find all distances from example to every sample in minoriry_sample
        for x in self.df[self.X_columns].values:
            distances.append(np.linalg.norm(x - example, ord=2))

        #select k neighbours
        predicted_index = np.argsort(distances)[1:k + 1]
        neighbours = self.df.iloc[predicted_index, :]
        good_neighbours = neighbours.loc[(neighbours.loc[:,'safe']==flag)&(neighbours.loc[:,self.target]==clas), self.X_columns + [self.target]]
        return good_neighbours