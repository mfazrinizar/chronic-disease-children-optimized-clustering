from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler


def get_scalers_map():
    return {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(quantile_range=(25, 75)),
        'MinMax': MinMaxScaler(),
        'MaxAbs': MaxAbsScaler()
    }
