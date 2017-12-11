library(stats)

setwd("./Downloads/DHFinalProject")
text <- read.table("./texts/berkeley.txt", fill = TRUE)
text

library(dplyr)
text_df <- data_frame(line = 1:4, text = text)

text_df
