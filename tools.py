import numpy as np

'''
This function calculates the annualized standard deviation of a dataframe.
@param dataframe: A pandas dataframe with the data to calculate the annualized standard deviation of.
@return: The annualized standard deviation of the dataframe.
'''
def annualized_std_dev(dataframe, data_type):
    data = dataframe[data_type]
    returns = data.pct_change()
    std_dev = returns.std()
    annualized_std_dev = std_dev * np.sqrt(252)
    return annualized_std_dev