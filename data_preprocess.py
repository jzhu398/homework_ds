# import packages from python
import pandas as pd
import category_encoders as ce

class data_preprocess:
    def __init__(self):
        pass

    def impute_process(feature):
        """
        This function is to help impute the missing data if needed

        Args:
            feature: pandas dataframe

        """
#############################################################################
#                                                                           # 
#      1.Clean/fill missing in numerical and categorical features           #
#                                                                           #
#############################################################################
        data = pd.DataFrame()
        categ_list = feature.select_dtypes(include=['object']).columns.tolist()
        num_list = feature.select_dtypes(exclude=['object']).columns.tolist()

        for num_features in num_list:
            mean_value = feature[num_features].mean()
            data[num_features] = feature[num_features].fillna(value=mean_value) 

        for categ_features in categ_list:
            data[categ_features] = feature[categ_features].fillna("-2")
        return data
    

    def encoding(train,test):
        """
        This function is to help categorical feature to be used in ML. It is 
        One-Hot encoding

        Args:
            data: pandas dataframe
            
        """
#############################################################################
#                                                                           #
#      2.One-hot encoding the categorical features                          #
#                                                                           #
#############################################################################
        categ_list = train.select_dtypes(include=['object']).columns.tolist()

        for categ in categ_list:
            encoder = ce.OneHotEncoder(use_cat_names=True)
            train = encoder.fit_transform(train)
            test = encoder.transform(test)
        return train, test




