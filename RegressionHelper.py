import pandas as pd
import statsmodels.api as sm 
from scipy.stats.mstats import zscore
def CreateEmotionColumns(valuations_arr, emotion):
    valuation_emotion_cols = []
    for i in range(len(valuations_arr)):
        column_name =emotion + valuations_arr[i]
        valuation_emotion_cols.append(column_name)
    return valuation_emotion_cols

def LinearRegression(emotion_concepts, df, prediction_columns, predicted_column):
    model_summaries = []
    for i in range(len(emotion_concepts)):
        emotion = emotion_concepts[i]
        emotion_val_columns = CreateEmotionColumns(prediction_columns, emotion)
        emotion_val_predict = emotion + predicted_column
        X = zscore(df[emotion_val_columns])
        y = zscore(df[emotion_val_predict])
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X) # make the predictions by the model

        # Print out the statistics
        model_summaries.append(model)
    return model_summaries

def create_df_of_betas_3(model_summaries, emo_concepts_regression):
    P_Val_Const,P_Val_Good,P_Val_Ideal,P_Val_Pleasant = [], [], [], []
    B_Val_Const, B_Val_Good, B_Val_Ideal, B_Val_Pleasant =  [], [], [], []
    r_squared = []
    for i in range(len(emo_concepts_regression)):
        P_Val_Const.append(model_summaries[i].pvalues[0])
        P_Val_Good.append(model_summaries[i].pvalues[1])
        P_Val_Ideal.append(model_summaries[i].pvalues[2])
        P_Val_Pleasant.append(model_summaries[i].pvalues[3])
        B_Val_Const.append(model_summaries[i].params[0])
        B_Val_Good.append(model_summaries[i].params[1])
        B_Val_Ideal.append(model_summaries[i].params[2])
        B_Val_Pleasant.append(model_summaries[i].params[3])
        r_squared.append(model_summaries[i].rsquared)

    return pd.DataFrame({
        'emotion':emo_concepts_regression,
        'P_Val_Const':P_Val_Const,
        'P_Val_Good':P_Val_Good,
        'P_Val_Ideal':P_Val_Ideal,
        'P_Val_Pleasant':P_Val_Pleasant,
        'B_Val_Const':B_Val_Const,
        'B_Val_Good':B_Val_Good,
        'B_Val_Ideal':B_Val_Ideal,
        'B_Val_Pleasant':B_Val_Pleasant,
        'R_squared':r_squared
    })

def create_df_of_betas_2(model_summaries, emo_concepts_regression):
    P_Val_Const,P_Val_Good,P_Val_Pleasant = [], [], []
    B_Val_Const, B_Val_Good,B_Val_Pleasant = [], [], []
    r_squared = []
    
    for i in range(len(emo_concepts_regression)):
        P_Val_Const.append(model_summaries[i].pvalues[0])
        P_Val_Good.append(model_summaries[i].pvalues[1])
        P_Val_Pleasant.append(model_summaries[i].pvalues[2])
        B_Val_Const.append(model_summaries[i].params[0])
        B_Val_Good.append(model_summaries[i].params[1])
        B_Val_Pleasant.append(model_summaries[i].params[2])
        r_squared.append(model_summaries[i].rsquared)

    return pd.DataFrame({
        'emotion':emo_concepts_regression,
        'P_Val_Const':P_Val_Const,
        'P_Val_Good':P_Val_Good,
        'P_Val_Pleasant':P_Val_Pleasant,
        'B_Val_Const':B_Val_Const,
        'B_Val_Good':B_Val_Good,
        'B_Val_Pleasant':B_Val_Pleasant,
        'R_squared':r_squared
    }) 

def create_df_of_betas_1(model_summaries, emo_concepts_regression):
    P_Val_Const, P_Val_Ideal, B_Val_Const, B_Val_Ideal, r_squared = [],[],[],[],[]
    for i in range(len(emo_concepts_regression)):
        P_Val_Const.append(model_summaries[i].pvalues[0])
        P_Val_Ideal.append(model_summaries[i].pvalues[1])
        B_Val_Const.append(model_summaries[i].params[0])
        B_Val_Ideal.append(model_summaries[i].params[1])
        r_squared.append(model_summaries[i].rsquared)

    return pd.DataFrame({
        'emotion':emo_concepts_regression,
        'P_Val_Const':P_Val_Const,
        'P_Val_Ideal':P_Val_Ideal,
        'B_Val_Const':B_Val_Const,
        'B_Val_Ideal':B_Val_Ideal,
        'R_squared':r_squared
    }) 