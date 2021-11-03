# import package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv(train_file, train_label_file, test_file):
    """
    Load the CSV form from train_features.csv, train_salaries.csv, test_features.csv

    Args:
        train_file, train_label_file, test_file: file path

    """
#################################################################################
#                                                                               #
#                             Read all three csv files                          #
#                                                                               #
#################################################################################
    trains = pd.read_csv(train_file, header = 0)
    train_label = pd.read_csv(train_label_file, header = 0)
    test = pd.read_csv(test_file, header = 0)
    
    join = pd.merge(left = trains, right = train_label, how = 'inner', on = 'jobId')
    
    train_labels = join.salary.values
    train = join.drop(labels=['jobId', 'salary','companyId'], axis=1)

    return train, train_labels, test


def plot_feature_importance(importance,names,model_type):
    """
    plot features importance
    Note:
        should consider permutation importance

    Args:
        importance: feature importance from the model
        names: column names
        model_type: str, model name

    """
#################################################################################
#                                                                               #
#                             Read all three csv files                          #
#                                                                               #
#################################################################################
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()