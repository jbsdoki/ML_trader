


def annualized_std_dev(dataframe):
    std_dev = dataframe.std()
    annualized_std_dev = std_dev * np.sqrt(252)
    return annualized_std_dev