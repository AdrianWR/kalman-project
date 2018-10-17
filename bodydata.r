#! /usr/bin/env Rscript

setwd("AnthropometricData")
dataframe <- read.csv("ANSUR II MALE Public.csv", header = TRUE, sep = ",")

chest <- dataframe[c("chestdepth", "chestbreadth")]
a <- (chest$chestbreadth/2)
b <- (chest$chestdepth/2)
rc_minor <- (chest$chestdepth**2)/(2*chest$chestbreadth)
rc_major <- (chest$chestbreadth**2)/(2*chest$chestdepth)

output <- paste('[ { "name" : "radius of curvature", "axis" : "minor", "mean" :', mean(rc_minor), ', "std" : ',sd(rc_minor), ', "data" : [',toString(rc_minor),'] } ,')
output <- paste(output, '{ "name" : "radius of curvature", "axis" : "major", "mean" : ', mean(rc_major), ', "std" : ',sd(rc_major), ', "data" : [',toString(rc_major),'] }, ')
output <- paste(output, '{ "name" : "semi-axis", "axis" : "major", "mean" : ', mean(a), ', "std" : ',sd(a), ', "data" : [',toString(a),'] }, ')
output <- paste(output, '{ "name" : "semi-axis", "axis" : "minor", "mean" : ', mean(b), ', "std" : ',sd(b), ', "data" : [',toString(b),'] } ]')

write(output, stdout())