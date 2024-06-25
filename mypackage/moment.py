# import
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# (1) mean var skew kurt
def get_moment(df_main):
    
    weighted_mean = np.sum(df_main["K"] * df_main["RND"]) / np.sum(df_main["RND"]) #F?
    
    weighted_variance = np.sum(df_main["RND"] * (df_main["K"] - weighted_mean)**2) / np.sum(df_main["RND"])

    weighted_std = np.sqrt(weighted_variance)

    weighted_skew = (np.sum(df_main["RND"] * (df_main["K"] - weighted_mean)**3) / 
                        np.sum(df_main["RND"])) / weighted_std**3
    excess_kurt = (np.sum(df_main["RND"] * (df_main["K"] - weighted_mean)**4) / 
                        np.sum(df_main["RND"])) / weighted_std**4 - 3     # 已經-3 !!
    
    
    print(f"mean: {round(weighted_mean)}, var: {round(weighted_variance)}")

    return weighted_mean, weighted_std, weighted_skew, excess_kurt
