data <- read.table("/home/hal9000/Repos/DEN1_GeneralizationAtRetrieval/rsc/freq_data.txt", header=TRUE)

model <- lm(discriminability ~ g_fr + g1_fr + g2_fr + ug_fr + ug1_fr + ug2_fr, data = data)

summary(model)
