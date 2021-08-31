library(lmerTest)
library(MuMIn)
library(effects)
library(ggplot2)
library(ezDiffusion)
library(ggpubr)
library(ggsignif)
library(plyr)


#Goodman, McClelland, Gibbs (1981)

df_goodman_table3 <- data.frame(Congruity=c("Appropriate", "Inappropriate", "Appropriate", "Inappropriate"),
                                PoS=c("Noun", "Noun", "Verb", "Verb"),
                                MRT=c(530, 547, 565, 585))



df_goodman_table1 <- data.frame(Target=c("Appropriate", "Inappropriate"),
                                MRT = c(530, 549),
                                Pc = c(1-0.03, 1 - 0.07),
                                MSe= c(1315.49, 1315.49))
df_goodman_table2 <- data.frame(Target=c("Appropriate", "Inappropriate"),
                                MRT=c(574, 589),
                                Pc=c(1 - 0.02, 1 - 0.03),
                                MSe=c(1398.02, 1398.02))

#Convert to ezDiffusion params

MRT_app <- (subset(df_goodman_table1, Target=="Appropriate")$MRT + subset(df_goodman_table2, Target=="Appropriate")$MRT)/2
MRT_inapp <- (subset(df_goodman_table1, Target=="Inappropriate")$MRT + subset(df_goodman_table2, Target=="Inappropriate")$MRT)/2
VRT_app <- ((subset(df_goodman_table1, Target=="Appropriate")$MSe + subset(df_goodman_table2, Target=="Appropriate")$MSe)/2)
VRT_inapp <- ((subset(df_goodman_table1, Target=="Inappropriate")$MSe + subset(df_goodman_table2, Target=="Inappropriate")$MSe)/2)
Pc_app <- (subset(df_goodman_table1, Target=="Appropriate")$Pc + subset(df_goodman_table2, Target=="Appropriate")$Pc)/2
Pc_inapp<- (subset(df_goodman_table1, Target=="Inappropriate")$Pc + subset(df_goodman_table2, Target=="Inappropriate")$Pc)/2

df_goodman_data <- data.frame(MRT = append(MRT_inapp, MRT_app)/1000, 
                              VRT = append(VRT_inapp, VRT_app)/(1000**2),
                              Pc = append(Pc_inapp, Pc_app))
#Fit Ez
s = 0.1
ef1 <- ezDiffusion(data = df_goodman_data, proportion_correct= "Pc", rt_variance = "VRT", rt_mean = "MRT")



##load model familiarity data across models

df <- read.csv('goodman_DEN.csv')
df$Familiarity <- ((df$Familiarity - mean(df$Familiarity) )/sd(df$Familiarity)) #our a priori estimate of mean drift
df["Model"] <- replicate(length(df$PoS), c("DEN"))

fam <- df$Familiarity
Cong<- df$Congruity
PoS <- df$PoS 
Item <- df$Item
Model<- df$Model


for (model in c("BSB", "pLAN")){
  df_model <- read.csv(paste("goodman_",model, ".csv",sep=""))
  df_model$Familiarity <- ((df_model$Familiarity - mean(df_model$Familiarity) )/sd(df_model$Familiarity)) #our a priori estimate of mean drift
  df_model["Model"] <- replicate(length(df_model$PoS), c(model))
  fam <- append(fam, df_model$Familiarity)
  Cong<-append(Cong, df_model$Congruity)
  PoS<-append(PoS, df_model$PoS)
  Item<-append(Item, df_model$Item)
  Model<-append(Model, df_model$Model)

}

df <- data.frame(Familiarity = fam,
                 Congruity = Cong,
                 PoS = PoS,
                 Item = Item, 
                 Model = Model)




model <- "DEN"
stat_model <- lmer(Familiarity ~ Congruity*PoS + (1|Item), data=subset(df, Model==model))

print(r.squaredGLMM(stat_model))


print(summary(stat_model))
marginal <- Effect(c("Congruity","PoS"), stat_model)
df_marginal <- data.frame(Target=marginal$x$Congruity, 
                              PoS =marginal$x$PoS,
                              Familiarity=marginal$fit,
                              Lower = marginal$lower,
                              Upper = marginal$upper)

fam <- marginal$fit 
Target <- marginal$x$Congruity 
PoS <- marginal$x$PoS 
fam_upper <- marginal$upper 
fam_lower <- marginal$lower 
Model <- replicate(length(marginal$fit), c(model))

#Transform marginal estimates to RTs using EZdiffusion params 

fam_app <- subset(df_marginal, Target =="Appropriate")$Familiarity
fam_app_upper<-subset(df_marginal, Target =="Appropriate")$Upper
fam_app_lower<-subset(df_marginal, Target =="Appropriate")$Lower 
PoS_app <- subset(df_marginal, Target =="Appropriate")$PoS

fam_inapp<- subset(df_marginal, Target =="Inappropriate")$Familiarity
fam_inapp_upper<-subset(df_marginal, Target =="Inappropriate")$Upper
fam_inapp_lower<-subset(df_marginal, Target =="Inappropriate")$Lower 
PoS_inapp<- subset(df_marginal, Target =="Inappropriate")$PoS


marginal_tr <- data.frame(Target=c("Appropriate", "Appropriate", "Inappropriate", "Inappropriate"), 
                              PoS=append(PoS_app, PoS_inapp), 
                              v=append(fam_app, fam_inapp),
                              v_upper=append(fam_app_upper, fam_inapp_upper),
                              v_lower=append(fam_app_lower, fam_inapp_lower),
                              y=append(-fam_app*ef1$a[1]/(s**2), -fam_inapp*ef1$a[2]/(s**2)),
                              y_upper = append(-fam_app_upper*ef1$a[1]/(s**2), -fam_inapp_upper*ef1$a[2]/(s**2)),
                              y_lower = append(-fam_app_lower*ef1$a[1]/(s**2), -fam_inapp_lower*ef1$a[2]/(s**2)),
                              MDT=append(ef1$a[1]/(2*fam_app)*((1 - exp(-fam_app*ef1$a[1]/(s**2)))/(1 + exp(-fam_app*ef1$a[1]/(s**2)))), 
                                         ef1$a[1]/(2*fam_inapp)*((1 - exp(-fam_inapp*ef1$a[1]/(s**2)))/(1 + exp(-fam_inapp*ef1$a[1]/(s**2))))),
                              MDT_upper=append(ef1$a[1]/(2*fam_app_upper)*((1 - exp(-fam_app_upper*ef1$a[1]/(s**2)))/(1 + exp(-fam_app_upper*ef1$a[1]/(s**2)))),
                                         ef1$a[1]/(2*fam_inapp_upper)*((1 - exp(-fam_inapp_upper*ef1$a[1]/(s**2)))/(1 + exp(-fam_inapp_upper*ef1$a[1]/(s**2))))),
                              MDT_lower=append(ef1$a[1]/(2*fam_app_lower)*((1 - exp(-fam_app_lower*ef1$a[1]/(s**2)))/(1 + exp(-fam_app_lower*ef1$a[1]/(s**2)))),
                                          ef1$a[1]/(2*fam_inapp_lower)*((1 - exp(-fam_inapp_lower*ef1$a[1]/(s**2)))/(1 + exp(-fam_inapp_lower*ef1$a[1]/(s**2))))),
                              t0 = c(ef1$t0[2], ef1$t0[2], ef1$t0[1], ef1$t0[1]))

#Final estimates
MDT_tr <- 1000*marginal_tr$MDT
MRT_tr <-  1000*(marginal_tr$MDT + marginal_tr$t0)
rt_upper_tr <- 1000*(marginal_tr$MDT_upper + marginal_tr$t0)
rt_lower_tr <- 1000*(marginal_tr$MDT_lower + marginal_tr$t0)
PoS_tr <- marginal_tr$PoS
Target_tr <- marginal_tr$Target
Model_tr <- replicate(length(MRT_tr), c(model))


for (model in c("pLAN", "BSB")){  
  stat_model <- lmer(Familiarity ~ Congruity*PoS + (1|Item), data=subset(df, Model==model))
  
  print(r.squaredGLMM(stat_model))
  
  
  print(summary(stat_model))
  marginal <- Effect(c("Congruity","PoS"), stat_model)
  df_marginal <- data.frame(Target=marginal$x$Congruity, 
                                PoS =marginal$x$PoS,
                                Familiarity=marginal$fit,
                                Lower = marginal$lower,
                                Upper = marginal$upper)
  fam <- append(fam, marginal$fit )
  Target <- append(Target, marginal$x$Congruity )
  PoS <- append(PoS, marginal$x$PoS )
  fam_upper <- append(fam_upper, marginal$upper )
  fam_lower <- append(fam_lower, marginal$lower )
  Model <- append(Model, replicate(length(marginal$fit), c(model)) )

  
  #Transform marginal estimates to RTs using EZdiffusion params 
  
  fam_app <- subset(df_marginal, Target =="Appropriate")$Familiarity
  fam_app_upper<-subset(df_marginal, Target =="Appropriate")$Upper
  fam_app_lower<-subset(df_marginal, Target =="Appropriate")$Lower 
  PoS_app <- subset(df_marginal, Target =="Appropriate")$PoS
  
  fam_inapp<- subset(df_marginal, Target =="Inappropriate")$Familiarity
  fam_inapp_upper<-subset(df_marginal, Target =="Inappropriate")$Upper
  fam_inapp_lower<-subset(df_marginal, Target =="Inappropriate")$Lower 
  PoS_inapp<- subset(df_marginal, Target =="Inappropriate")$PoS
  
  
  marginal_tr <- data.frame(Target=c("Appropriate", "Appropriate", "Inappropriate", "Inappropriate"), 
                                PoS=append(PoS_app, PoS_inapp), 
                                v=append(fam_app, fam_inapp),
                                v_upper=append(fam_app_upper, fam_inapp_upper),
                                v_lower=append(fam_app_lower, fam_inapp_lower),
                                y=append(-fam_app*ef1$a[1]/(s**2), -fam_inapp*ef1$a[2]/(s**2)),
                                y_upper = append(-fam_app_upper*ef1$a[1]/(s**2), -fam_inapp_upper*ef1$a[2]/(s**2)),
                                y_lower = append(-fam_app_lower*ef1$a[1]/(s**2), -fam_inapp_lower*ef1$a[2]/(s**2)),
                                MDT=append(ef1$a[1]/(2*fam_app)*((1 - exp(-fam_app*ef1$a[1]/(s**2)))/(1 + exp(-fam_app*ef1$a[1]/(s**2)))), 
                                           ef1$a[1]/(2*fam_inapp)*((1 - exp(-fam_inapp*ef1$a[1]/(s**2)))/(1 + exp(-fam_inapp*ef1$a[1]/(s**2))))),
                                MDT_upper=append(ef1$a[1]/(2*fam_app_upper)*((1 - exp(-fam_app_upper*ef1$a[1]/(s**2)))/(1 + exp(-fam_app_upper*ef1$a[1]/(s**2)))),
                                           ef1$a[1]/(2*fam_inapp_upper)*((1 - exp(-fam_inapp_upper*ef1$a[1]/(s**2)))/(1 + exp(-fam_inapp_upper*ef1$a[1]/(s**2))))),
                                MDT_lower=append(ef1$a[1]/(2*fam_app_lower)*((1 - exp(-fam_app_lower*ef1$a[1]/(s**2)))/(1 + exp(-fam_app_lower*ef1$a[1]/(s**2)))),
                                            ef1$a[1]/(2*fam_inapp_lower)*((1 - exp(-fam_inapp_lower*ef1$a[1]/(s**2)))/(1 + exp(-fam_inapp_lower*ef1$a[1]/(s**2))))),
                                t0 = c(ef1$t0[2], ef1$t0[2], ef1$t0[1], ef1$t0[1]))
  
  #Final estimates
  MDT_tr <- append(MDT_tr, 1000*marginal_tr$MDT)
  MRT_tr <- append(MRT_tr, 1000*(marginal_tr$MDT + marginal_tr$t0))
  rt_upper_tr <- append(rt_upper_tr, 1000*(marginal_tr$MDT_upper + marginal_tr$t0))
  rt_lower_tr <- append(rt_lower_tr, 1000*(marginal_tr$MDT_lower + marginal_tr$t0))
  PoS_tr <- append(PoS_tr, marginal_tr$PoS)
  Target_tr <- append(Target_tr, marginal_tr$Target)
  Model_tr <- append(Model_tr, replicate(length(marginal_tr$PoS), c(model)))

}

marginal <- data.frame(Congruity = Target,
                       PoS = PoS,
                       Upper = fam_upper,
                       Lower = fam_lower,
                       Model = Model_tr,
                       zFamiliarity = fam, 
                       MRT = MRT_tr,
                       MDT = MDT_tr,
                       rt_upper = rt_upper_tr,
                       rt_lower = rt_lower_tr)


drift_plot_DEN <- ggplot(subset(marginal, Model == "DEN"), aes(x=PoS, y=zFamiliarity, fill=Congruity)) +
      geom_bar(position=position_dodge(), stat='identity', color='black', show.legend=FALSE) +
      coord_cartesian(ylim=c(-0.9,0.9)) + geom_errorbar(aes(ymin=Lower, ymax=Upper), width=.2, position=position_dodge(.9)) +  
        scale_fill_manual(values=c("red", "blue")) + 
        geom_signif(annotation=c("***"), comparisons=list(c("Noun", "Verb")), y_position = max(subset(marginal, Model == "DEN")$zFamiliarity)+0.15) +
        geom_signif(y_position = c(max(subset(marginal, Model == "DEN" & PoS == "Noun")$zFamiliarity)+0.1, 
                                   max(subset(marginal, Model == "DEN" & PoS == "Verb")$zFamiliarity)+0.1), xmin = c(0.8,1.8),xmax = c(1.2,2.2), annotation = c("***","***"))



drift_plot_pLAN <- ggplot(subset(marginal, Model == "pLAN"), aes(x=PoS, y=zFamiliarity, fill=Congruity)) +
      geom_bar(position=position_dodge(), stat='identity', color='black', show.legend=FALSE) +
      coord_cartesian(ylim=c(-0.9,0.9)) + geom_errorbar(aes(ymin=Lower, ymax=Upper), width=.2, position=position_dodge(.9)) +  
        scale_fill_manual(values=c("red", "blue")) +
        geom_signif(annotation=c("***"), comparisons=list(c("Noun", "Verb")), y_position = max(subset(marginal, Model == "DEN")$zFamiliarity)+0.1) +
        geom_signif(y_position = c(max(subset(marginal, Model == "pLAN" & PoS == "Noun")$zFamiliarity)+0.1,
                           max(subset(marginal, Model == "pLAN" & PoS == "Verb")$zFamiliarity)+0.1), xmin = c(0.8,1.8),xmax = c(1.2,2.2), annotation = c("***","***"))


drift_plot_BSB <- ggplot(subset(marginal, Model == "BSB"), aes(x=PoS, y=zFamiliarity, fill=Congruity)) +
      geom_bar(position=position_dodge(), stat='identity', color='black') +
      coord_cartesian(ylim=c(-0.9,0.9)) + geom_errorbar(aes(ymin=Lower, ymax=Upper), width=.2, position=position_dodge(.9)) +  
        scale_fill_manual(values=c("red", "blue")) +
        geom_signif(y_position =c(max(subset(marginal, Model == "BSB" & PoS == "Noun")$zFamiliarity)+0.1), xmin = c(0.8),xmax = c(1.2), annotation = c("*"))











goodman_plot <- ggplot(df_goodman_table3, aes(x=PoS, y=MRT, fill=Congruity)) +
                geom_bar(position=position_dodge(), stat='identity', color="black", show.legend = TRUE) + 
                coord_cartesian(ylim=c(500,600)) +
                scale_fill_manual(values=c("red", "blue")) + 
                geom_signif(annotation=c("*"), comparisons=list(c("Noun", "Verb")), y_position = max(df_goodman_table3$MRT)+7.5) + 
                geom_signif(y_position = c(max(subset(df_goodman_table3, PoS == "Noun")$MRT)+2.5, 
                                           max(subset(df_goodman_table3, PoS == "Verb")$MRT)+2.5), xmin = c(0.8,1.8),xmax = c(1.2,2.2), annotation = c("*","*"))

rt_plot_DEN <- ggplot(subset(marginal, Model == "DEN"), aes(x=PoS, y=MRT, fill=Congruity)) + 
      geom_bar(position=position_dodge(), stat='identity', color="black", show.legend = FALSE) + 
      coord_cartesian(ylim=c(500,600)) + geom_errorbar(aes(ymin=rt_lower, ymax=rt_upper), width=.2, position=position_dodge(.9)) + 
      scale_fill_manual(values=c("red", "blue"))


rt_plot_pLAN <- ggplot(subset(marginal, Model == "pLAN"), aes(x=PoS, y=MRT, fill=Congruity)) + 
      geom_bar(position=position_dodge(), stat='identity', color="black", show.legend = FALSE) + 
      coord_cartesian(ylim=c(500,600)) + geom_errorbar(aes(ymin=rt_lower, ymax=rt_upper), width=.2, position=position_dodge(.9)) + 
      scale_fill_manual(values=c("red", "blue"))



print(ggarrange( drift_plot_BSB + rremove("xlab"), drift_plot_DEN + rremove("ylab") + rremove("xlab"), drift_plot_pLAN + rremove("xlab") + rremove("ylab"), goodman_plot, rt_plot_DEN + rremove("ylab"), rt_plot_pLAN + rremove("ylab"),
                labels=c("BSB","DEN","pLAN", "Goodman et al. (1981)", "DEN + EZdiffusion", "pLAN + EZdiffusion"),
                ncol=3, nrow=2, common.legend = TRUE, legend="top"))



























































