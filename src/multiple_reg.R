data <- read.table("/home/hal9000/Repos/DEN1_GeneralizationAtRetrieval/rsc/freq_data.txt", header=TRUE)


centered <- data.frame(scale(data$Discriminability, center=TRUE, scale=FALSE), 
                       scale(data$MedianRT, center=TRUE, scale=FALSE),
                       scale(data$g_fr, center=TRUE, scale=FALSE),
                       scale(data$ug_fr, center=TRUE, scale=FALSE),
                       scale(data$g1_fr, center=TRUE, scale=FALSE),
                       scale(data$g2_fr, center=TRUE, scale=FALSE),
                       scale(data$ug1_fr, center=TRUE, scale=FALSE),
                       scale(data$ug2_fr, center=TRUE, scale=FALSE))

colnames(centered) <- c("d", "RT", "fG", "fUG", "fG1", "fG2", "fUG1", "fUG2")




freqModel <- lm(RT ~ fG + fUG + fG1 + fG2 + fUG1 + fUG2, data=centered)
summary(freqModel)

freqVsD <- lm(RT ~ fG + d, data=centered)
summary(freqVsD)
