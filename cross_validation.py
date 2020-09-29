# Import pandas and model_Selection module of Scikit-learn
import pandas as pd 
from sklearn import model_selection

class cross_validation :
    """
    This class will divide dataset into two part one is for training and one is for validation
    """
    def __init__(self , df , target = None , n = 5):
        '''
        df : Dataframe of the dataset
        target : target value of the dataset
        n : number of split
        '''
        self.df = df
        self.target = target
        self.n = n

    def kfold(self):
        # we create a new column called kfold and fill it with -1
        self.df['kfold'] = -1

        # the next step is to randomize the rows of the data
        self.df = self.df.sample(frac = 1).reset_index(drop = True)

        #initiate the kfold class from the module_selection module
        kf = model_selection.KFold(n_splits= self.n)

        # fill the new kfold column
        for fold , (trn_ , val_) in enumerate(kf.split(X = self.df)):
            self.df.loc[val_ , 'kfold'] = fold 

        # return the new df with kfold column
        return self.df

    def StratifiedKFold (self):
        # we create a new column called kfold and fill it with - 1
        self.df['kfold'] = -1

        # fetch the targets
        y = self.df[self.target].values

        #intiate the kfold class from model_selection module
        kf = model_selection.StratifiedKFold(n_splits= self.n)

        # fill the new kfold column
        for f, (t_ , v_) in enumerate(kf.split( X = self.df , y= y)):
            self.df.loc[v_ , 'kfold'] = f

        # return the new df with kfold column
        return self.df
