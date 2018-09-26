setwd("AnthropometricData")
dataframe <- read.csv("ANSUR II MALE Public.csv", header = TRUE, sep = ",")

chest <- dataframe[c("chestdepth", "chestbreadth")]
rc_minor_axis <- (chest$chestdepth**2)/(2*chest$chestbreadth)
rc_major_axis <- (chest$chestbreadth**2)/(2*chest$chestdepth)

mean(rc_minor_axis)