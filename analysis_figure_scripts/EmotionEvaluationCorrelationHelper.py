import pandas as pd
import numpy as np
from scipy import stats
import math
def getPairwiseCorrelations(df,value_dimension_1, value_dimension_2):
    columns = df.columns
    
    emotion_concepts = [i.split(value_dimension_1)[0] for i in columns if len(i.split(value_dimension_1))>1]
    pairwise_corr = []
    
    for emotion in emotion_concepts:
        good_column = emotion + str(value_dimension_1)
        pleasant_column = emotion + str(value_dimension_2)
        corr_matrix = df[[good_column, pleasant_column]].corr()
        pairwise_corr.append(corr_matrix.iloc[0,1])
            
    return pd.DataFrame(
              data={'emotions' : emotion_concepts,
                    'corr_val_dim':pairwise_corr})

def getColumnNameTriples(start_col, df, end_offset):
    rows, cols = df.shape
    end_col = start_col+4
    
    
    cols -=end_offset
    
    columns = df.columns
    
    emotion_concepts_names = []
    GoodPleasant_corr = []
    PleasantIdeal_corr = []
    GoodIdeal_corr = []

    while (end_col <= cols):
        good_pleasant_cols = columns[start_col: end_col]
        emotion_concept = good_pleasant_cols[1].split('Pl')[0]
        emotion_concepts_names.append(emotion_concept)
        corr_matrix = df[good_pleasant_cols].corr()
        GoodPleasant_corr.append(corr_matrix.iloc[0,1])
        GoodIdeal_corr.append(corr_matrix.iloc[0,2])
        PleasantIdeal_corr.append(corr_matrix.iloc[1,2])
        start_col +=4
        end_col +=4
        
    return pd.DataFrame(
              data={'emotions' : emotion_concepts_names,
                    'GoodPleasant_Corr':GoodPleasant_corr,
                    'GoodIdeal_corr':GoodIdeal_corr,
                    'PleasantIdeal_corr':PleasantIdeal_corr
                   })
#consolide ranks of emotion to graph them
def createRanksDictionary(study_1, study_2): #method assumes same emotions
    emotion_ranks_dict = {}
    ranks_study_1 = []
    ranks_study_2 = []
    for emotion in study_1['emotions']:
        study_1_rank = study_1[study_1['emotions']==emotion].index.values[0]
        study_2_rank = study_2[study_2['emotions']==emotion].index.values[0]
        emotion_ranks_dict[emotion]=[study_1_rank, study_2_rank]
        ranks_study_1.append(study_1_rank)
        ranks_study_2.append(study_2_rank)
    return emotion_ranks_dict, ranks_study_1, ranks_study_2

def CreateRValueDictionaryAcrossStudies(dictionary, study_df, index):
    for emotion in study_df['emotions']:
        current_word_r_value = study_df.loc[study_df['emotions'] == emotion]['corr_val_dim'].values[0]
        if emotion in dictionary:
            dictionary[emotion][index]=current_word_r_value
    return dictionary


#http://danielnee.com/2015/01/random-permutation-tests/
def randomPermutationSpearman(ranks_study_1, ranks_study_2):
    spearman_ranks = stats.spearmanr(ranks_study_1,ranks_study_2)
    perms = np.zeros(1000)
    for i in range(0,len(perms)):
        study_1_rand_order = np.random.permutation(ranks_study_1)
        study_2_rand_order = np.random.permutation(ranks_study_2)
        perms[i]=stats.spearmanr(study_1_rand_order, study_2_rand_order).correlation
    perms.sort()
    bigger_smaller = sum([i > spearman_ranks for i in perms])
    return ((bigger_smaller[0])+1)/(len(perms)+1)
#to do test this function
def dependent_samples_corr_signifigance_test(N,
                                             p_jk,#variable 1 of interest
                                             p_jh,
                                             p_jm,
                                             p_km,
                                             p_kh,
                                             p_hm #variable 2 of interest
                                            ):
    
    
    covar_jk_hm = ((p_jh-p_jk*p_kh)*(p_km-p_kh*p_hm)
                      +(p_jm-p_jh*p_hm)*(p_kh-p_jk*p_jh) +
                      (p_jh-p_jm*p_hm)*(p_km-p_jk*p_jm) +
                      (p_jm-p_jk*p_km)*(p_kh-p_km*p_hm))/2
    
    
    sample_estimate = covar_jk_hm /((1-p_jk**2)*(1-p_hm**2))
    Z_jk= np.arctanh(p_jk)
    Z_hm = np.arctanh(p_hm)
    return (N-3)**(.5)*(Z_jk-Z_hm)*(2-2*sample_estimate)**(-.5)

def ratingColumn(column_name,string_1, string_2):
    return len(column_name.split(string_1))>1 or len(column_name.split(string_2))>1
#to do write tests for this function
def calculateZScore(corr_matrix,
                    N,
                    valuation_col_1,
                    valuation_col_2,
                    emotion_2,
                    emotion_1):
    #get column/row names for each emotion
        j = emotion_1+str(valuation_col_1)
        k = emotion_1+str(valuation_col_2)
        h = emotion_2+str(valuation_col_1)
        m = emotion_2+str(valuation_col_2)

        #get correlations between individual components
        emotion_1_good_pleasant               = corr_matrix[j][k]
        emotion_1_good_emotion_2_good         = corr_matrix[j][h]
        emotion_1_good_emotion_2_pleasant     = corr_matrix[j][m]
        emotion_1_pleasant_emotion_2_pleasant = corr_matrix[k][m]
        emotion_1_pleasant_emotion_2_good     = corr_matrix[k][h]
        emotion_2_good_pleasant               = corr_matrix[h][m]

        Z_star = dependent_samples_corr_signifigance_test(N,
                                                 emotion_1_good_pleasant,
                                                 emotion_1_good_emotion_2_good,
                                                 emotion_1_good_emotion_2_pleasant,
                                                 emotion_1_pleasant_emotion_2_pleasant,
                                                 emotion_1_pleasant_emotion_2_good,
                                                 emotion_2_good_pleasant)
        return Z_star
#to do write tests for this function
def createZScoreDf(study_df, emotion_concepts, valuation_col_1, valuation_col_2):
    study_cols = study_df.columns
    #get all emotion concepts valuations col
    emotion_concepts_across_dimension =[i for i in study_cols if ratingColumn(i,valuation_col_1,valuation_col_2)] 
    corr_matrix = study_df[emotion_concepts_across_dimension].corr()
    
    N = len(study_df) #sample size
    basic_diff_map = {}    
    
    #iterate over all rows
    for emotion_1 in emotion_concepts:
        comparisons = []
        #iterate overall columns
        for emotion_2 in emotion_concepts:
            z_score = calculateZScore(corr_matrix,
                                      N,
                                      valuation_col_1,
                                      valuation_col_2,
                                      emotion_1,
                                      emotion_2)
            comparisons.append(z_score)#inner for loop append z score
            
        basic_diff_map[emotion_1]=comparisons#outer for loop append column of z scores
    return pd.DataFrame(data=basic_diff_map, index=emotion_concepts)

 #get T-test values for an emotion   
def getTTestValues(df, val_dim_1, val_dim_2):
    #get emotion concepts
    emotion_concepts = [i.split(val_dim_1)[0] for i in df.columns if len(i.split(val_dim_1))>1]
    t_vals = []
    p_vals = []
    val_dim_1_mean = []
    val_dim_1_std = []
    val_dim_2_mean = []
    val_dim_2_std = []
    mean_difference = []
    for emotion in emotion_concepts:
        emotion_good = emotion + str(val_dim_1)
        emotion_pleasant = emotion+str(val_dim_2)
        #drop_nas_for_each_col = df[[emotion_pleasant,emotion_good]].dropna()
        emotion_good = df[emotion_good].dropna() #drop_nas_for_each_col[emotion_good]
        emotion_pleasant = df[emotion_pleasant].dropna() #drop_nas_for_each_col[emotion_pleasant]
        sqrt_length = math.sqrt(len(emotion_good))
        val_dim_1_mean.append(emotion_good.mean())
        val_dim_1_std.append(emotion_good.std()/sqrt_length)
        val_dim_2_mean.append(emotion_pleasant.mean())
        val_dim_2_std.append(emotion_pleasant.std()/sqrt_length)
        
        tval, pval =stats.ttest_ind(emotion_good,emotion_pleasant)
        t_vals.append(tval)
        p_vals.append(pval)
        
    return pd.DataFrame(data = {
        'emotion':emotion_concepts,
        val_dim_1 + '_mean': val_dim_1_mean,
        val_dim_1 + '_std': val_dim_1_std,
        val_dim_2 + '_mean': val_dim_2_mean,
        val_dim_2 + '_std': val_dim_2_std,
        't_vals':t_vals,
        'pvals': p_vals
    })