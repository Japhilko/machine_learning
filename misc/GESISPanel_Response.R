# Jan-Philipp Kolb
# Fri Sep 28 11:27:43 2018

library(lattice)


## ------------------------------------------------------------------------
wave <- "fb"

## ------------------------------------------------------------------------
wavedatapath <- "J:/Work/GESISPanel_DATA/01_post_processing/c01/f_2018/fb/02_master/data/STATA14/"
main_path <- "J:/Work/GESISPanel_SQA/03_Forschung/jsm_statistical_learning/"
data_path <- paste0(main_path,"data/")
graph_path <- paste0(main_path,"figure/")

## ------------------------------------------------------------------------
setwd(wavedatapath)
dat <- readstata13::read.dta13("fb_master_20180814_COMPLETE.dta",convert.factors = F)

## ------------------------------------------------------------------------
ncol(dat)

## ------------------------------------------------------------------------
indvar_aapor <- grep("za006a",colnames(dat))

colnames(dat)[indvar_aapor]

## ------------------------------------------------------------------------
waves <- paste0(rep(letters[1:6],each=6,),rep(letters[1:6],6))
waves <- waves[-which(waves%in%c("ad","ae","af","fc","fd","fe","ff"))]

G_response_list <- list()
for (i in 1:length(waves)){
  ind_aapor <- which(colnames(dat)==paste0(waves[i],"za006a"))
  respvar <- dat[,ind_aapor]
  respvar1 <- respvar[respvar!="-22"] # not in panel
  G_response <- rep(0,length(respvar1))
  G_response[respvar1%in%c("211","212","319","21121","211221")] <- 1
  G_response_list[[i]] <- G_response
}

sumtab_resp <- lapply(G_response_list,table)

sumtab_resp2 <- data.frame(do.call(rbind, sumtab_resp))

# sumtab_resp2$wave <- waves

## ------------------------------------------------------------------------
table(dat$D_response)



## ------------------------------------------------------------------------
# Missing Patterns

list_respvar <- list()
for (i in 1:length(waves)){
  ind_aapor <- which(colnames(dat)==paste0(waves[i],"za006a"))
  respvar <- dat[,ind_aapor]
  respvar1 <- respvar[respvar!="-22"] # not in panel
  respvar1 <- factor(respvar1,levels = c("-11","11","12","212","232","319","331","3253","3311","21121","21131" ,"211211", "211212", "211221"))
  list_respvar[[i]] <- table(respvar1)
}

tab_respvar <- do.call(rbind, list_respvar)


plot(tab_respvar[,2],type="l")
plot(tab_respvar)


response <- tab_respvar[,2]
names(response) <- waves
barchart(response,col="royalblue")

#-----------------------------------------#


response <- sumtab_resp2[,2]
names(response) <- waves
barchart(response,col="royalblue")


library(ggplot2)
# Basic barplot

df_repsonse <- data.frame(nonresponse=response,waves)

p<-ggplot(data=df_repsonse, aes(x=waves, y=nonresponse)) +
  geom_bar(stat="identity",fill="royalblue",col="orange")

setwd(graph_path)
pdf("bar_nonresponse_waves.pdf")
  p
dev.off()


## ------------------------------------------------------------------------
# Die Daten zusammen bringen

cnames_tresp <- colnames(tab_respvar)

# restab <- data.frame(waves,tab_respvar)
# colnames(restab) <- c("wave",cnames_tresp)

restab2 <- data.frame(waves,tab_respvar,sumtab_resp2)


setwd(data_path)
save(restab2,file="response_tab.RData")
