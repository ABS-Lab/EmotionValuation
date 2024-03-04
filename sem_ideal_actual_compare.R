library(lavaan)
library(dplyr)

# Read data
df <- read.csv("/data/study4_olr_formatted.csv", sep = ',')

# List of emotions
emo_concepts_regression <- c('Shame', 'Fear', 'Calm', 'Jealousy', 'Miserable', 'Lust', 
                             'Disgust', 'Guilt', 'Pride', 'Excitement', 'Sadness', 'Happy', 'Love')
aggregated_results <- data.frame()
# Loop over each emotion
for (emotion in emo_concepts_regression) {
  eval_col <- paste(emotion, "Good", sep = "")
  actual_col <- paste(emotion, "Actual", sep = "")
  ideal_col <- paste(emotion, "Ideal", sep = "")
  pleasant_col <- paste(emotion, "Pleasant", sep = "")
  
  # Subset and standardize the data
  emo_df <- df %>% select(eval_col, actual_col, ideal_col, pleasant_col) %>% na.omit() %>% scale()
  
  # Define and fit the models
  model1_syntax <- paste(ideal_col, "~", eval_col, "+", pleasant_col,"\n",
                         actual_col, "~", ideal_col)
  fit1 <- sem(model1_syntax, data = emo_df)
  
  model2_syntax <- paste(actual_col, "~", eval_col, "+", pleasant_col,"\n",
                         ideal_col, "~", actual_col)
  fit2 <- sem(model2_syntax, data = emo_df)
  

  emotion
  # Extract and save the model summaries
  fit_stats <- data.frame(
    Emotion = c(emotion,emotion),
    Model = c("Model 1", "Model 2"),
    AIC = c(AIC(fit1), AIC(fit2)),
    BIC = c(BIC(fit1), BIC(fit2)),
    CFI = c(fitMeasures(fit1, "cfi"), fitMeasures(fit2, "cfi")),
    TLI = c(fitMeasures(fit1, "tli"), fitMeasures(fit2, "tli")),
    ChiSquare = c(fitMeasures(fit1, "chisq"), fitMeasures(fit2, "chisq")),
    PValue = c(fitMeasures(fit1, "pvalue"), fitMeasures(fit2, "pvalue"))
  )
  
  #write.csv(fit_stats, file = paste(emotion, "_model_summaries.csv", sep = ""))
  aggregated_results <- rbind(aggregated_results, fit_stats)
  
}

write.csv(aggregated_results, file ="aggregated_model_summaries.csv")
