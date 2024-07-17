library(WGCNA)
allowWGCNAThreads()
ALLOW_WGCNA_THREADS=60
args<-commandArgs(T)
powers1=c(seq(1,10,by=1),seq(12,30,by=2))
datExpr=read.table(args[1],sep="\t",row.names=1,header=T,check.names=F)
datExpr = t(datExpr)
gsg <- goodSamplesGenes(datExpr, verbose = 3)
if (!gsg$allOK) {
  # 异常的基因
  if (sum(!gsg$goodGenes)>0) 
    printFlush(paste("Removing genes:", 
                     paste(names(datExpr)[!gsg$goodGenes], collapse = ",")));
  # 异常的样本
  if (sum(!gsg$goodSamples)>0) 
    printFlush(paste("Removing samples:", 
                     paste(rownames(datExpr)[!gsg$goodSamples], collapse = ",")));
  # 删除异常的样本和基因
  datExpr = datExpr[gsg$goodSamples, gsg$goodGenes]
}
sft=pickSoftThreshold(datExpr, powerVector=powers1)
RpowerTable=sft[[2]]
cex1=0.9
pdf(file=args[2])
par(mfrow=c(1,2))
plot(RpowerTable[,1], -sign(RpowerTable[,3])*RpowerTable[,2],xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",,main = paste("Scale independence"))
text(RpowerTable[,1], -sign(RpowerTable[,3])*RpowerTable[,2], labels=powers1,cex=cex1,col="red")
abline(h=0.85,col="red")
plot(RpowerTable[,1], RpowerTable[,5],xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",main = paste("Mean connectivity"))
text(RpowerTable[,1], RpowerTable[,5], labels=powers1, cex=cex1,col="red")
dev.off()
beta1=sft[[1]]
ADJ= adjacency(datExpr,power=beta1)
vis=exportNetworkToCytoscape(ADJ,edgeFile=args[3],threshold = 0.1)
