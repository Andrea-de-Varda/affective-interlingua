# visualize set intersections
require(pacman)
p_load("reticulate")
p_load("SuperExactTest")
p_load(dplyr)

getwd()
setwd("/path/to/probing")

      #########
####### mBERT #######
      #########

###########
# VALENCE #
###########

n <- 100
s_100_mbert_valence <- list()

s_100_mbert_valence$German     <- read.csv("german/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Dutch      <- read.csv("dutch/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$French     <- read.csv("french/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Italian    <- read.csv("italian/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Portuguese <- read.csv("portuguese/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Spanish    <- read.csv("spanish/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Turkish    <- read.csv("turkish/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Chinese    <- read.csv("chinese/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$English    <- read.csv("english/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Indonesian    <- read.csv("indonesian/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Polish    <- read.csv("polish/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Greek    <- read.csv("greek/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_valence$Croatian    <- read.csv("croatian/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]

total <- 768
res_mbert_valence=supertest(s_100_mbert_valence, n=total)
plot(res_mbert_valence, sort.by="size", margin=c(2.8,2.8,2.8,8.9), 
     color.scale.pos=c(2.09,0.38), legend.pos=c(2.09,0.16), 
     keep.empty.intersections=T, 
     degree=c(2), minMinusLog10PValue=0,maxMinusLog10PValue=8) # pairs
#dev.off()
a <- summary(res_mbert_valence)
options(max.print=1000000)
sink("superset_valence_normalized.txt")
print(a)
sink()

###########
# AROUSAL #
###########

s_100_mbert_arousal <- list()

s_100_mbert_arousal$German     <- read.csv("german/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Dutch      <- read.csv("dutch/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$French     <- read.csv("french/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Italian    <- read.csv("italian/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Portuguese <- read.csv("portuguese/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Spanish    <- read.csv("spanish/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Turkish    <- read.csv("turkish/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Chinese    <- read.csv("chinese/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$English    <- read.csv("english/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Indonesian    <- read.csv("indonesian/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Polish    <- read.csv("polish/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Greek    <- read.csv("greek/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_mbert_arousal$Croatian    <- read.csv("croatian/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]

res_mbert_arousal=supertest(s_100_mbert_arousal, n=total)
#jpeg("plots_new/superset_mbert_valence.jpg", width = 747, height = 630, quality=100, res=300)
plot(res_mbert_arousal, sort.by="size", margin=c(2.8,2.8,2.8,8.9), 
     color.scale.pos=c(2.09,0.38), legend.pos=c(2.09,0.16), keep.empty.intersections=T, 
     degree=c(2), minMinusLog10PValue=0,maxMinusLog10PValue=8) # pairs
#dev.off()

b <- summary(res_mbert_arousal)
options(max.print=1000000)
sink("superset_arousal_normalized.txt")
print(b)
sink()




#########
####### XLM-R #######
#########

###########
# VALENCE #
###########

s_100_XLM_valence <- list()

s_100_XLM_valence$German     <- read.csv("german_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Dutch      <- read.csv("dutch_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$French     <- read.csv("french_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Italian    <- read.csv("italian_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Portuguese <- read.csv("portuguese_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Spanish    <- read.csv("spanish_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Turkish    <- read.csv("turkish_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Chinese    <- read.csv("chinese_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$English    <- read.csv("english_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Indonesian    <- read.csv("indonesian_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Polish    <- read.csv("polish_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Greek    <- read.csv("greek_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_valence$Croatian    <- read.csv("croatian_XLM-R/singlelayer_valence_ordering_normalized.csv")$neuron[1:n]

res_XLM_valence=supertest(s_100_XLM_valence, n=total)
#jpeg("plots_new/superset_xlm_valence.jpg")
plot(res_XLM_valence, sort.by="size", margin=c(2.8,2.8,2.8,8.9), 
     color.scale.pos=c(2.09,0.38), legend.pos=c(2.09,0.16), keep.empty.intersections=F, 
     degree=c(2), minMinusLog10PValue=0,maxMinusLog10PValue=8) # pairs
#dev.off()
a <- summary(res_XLM_valence)
options(max.print=1000000)
sink("superset_valence_xlm_normalized.txt")
print(a)
sink()

###########
# AROUSAL #
###########

s_100_XLM_arousal <- list()

s_100_XLM_arousal$German     <- read.csv("german_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Dutch      <- read.csv("dutch_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$French     <- read.csv("french_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Italian    <- read.csv("italian_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Portuguese <- read.csv("portuguese_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Spanish    <- read.csv("spanish_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Turkish    <- read.csv("turkish_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Chinese    <- read.csv("chinese_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$English    <- read.csv("english_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Indonesian    <- read.csv("indonesian_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Polish    <- read.csv("polish_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Greek    <- read.csv("greek_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]
s_100_XLM_arousal$Croatian    <- read.csv("croatian_XLM-R/singlelayer_arousal_ordering_normalized.csv")$neuron[1:n]

res_XLM_arousal=supertest(s_100_XLM_arousal, n=total)
#jpeg("plots_new/superset_xlm_arousal.jpg")
plot(res_XLM_arousal, sort.by="size", margin=c(2.8,2.8,2.8,8.9), 
     color.scale.pos=c(1.09,0.38), legend.pos=c(1.07,0.16), keep.empty.intersections=T, 
     degree=c(2), minMinusLog10PValue=0,maxMinusLog10PValue=8) # pairs
#dev.off()

b <- summary(res_XLM_arousal)
sink("superset_arousal_xlm_normalized.txt")
options(max.print=1000000)
print(b)
sink()
