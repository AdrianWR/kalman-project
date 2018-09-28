#! /usr/bin/env Rscript

setwd("AnthropometricData")
dataframe <- read.csv("ANSUR II MALE Public.csv", header = TRUE, sep = ",")

chest <- dataframe[c("chestdepth", "chestbreadth")]
rc_minor <- (chest$chestdepth**2)/(2*chest$chestbreadth)
rc_major <- (chest$chestbreadth**2)/(2*chest$chestdepth)

output <- paste('[ { "axis" : "minor", "mean" :', mean(rc_minor), ', "std" : ',sd(rc_minor), ', "data" : [',toString(rc_minor),'] } ,')
output <- paste(output, '{ "axis" : "major", "mean" : ', mean(rc_major), ', "std" : ',sd(rc_major), ', "data" : [',toString(rc_major),'] } ]')

write(output, stdout())