---
title: "Clustering"
author: "Jan-Philipp Kolb and Alexander Murray-Watters"
date: "18 Januar 2019"
output: 
  slidy_presentation: 
    keep_md: yes
---






## Resources





- [Package `kknn`](https://cran.r-project.org/web/packages/kknn/kknn.pdf)






## [Geographic clustering of UK cities](https://www.r-bloggers.com/geographic-clustering-of-uk-cities/)

Animated example: 
https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68


## Exercise: Kmeans

Apply kmeans to to the `iris` dataset with 2, 3, and 4
clusters. Produce three scatter plots, with the points colored
according to cluster assignment.


## hdbscan

A fairly new alternative to kmeans, hdbscan does not require you to
specify the number of categories to be assigned. It only requires a
decision as to the minimum number of points needed to be included in a
cluster. This minimum number acts as a smoothing parameter (such as a
density bandwidth parameter or a histograms bin/bar width), with lower
values finding more clusters. Other advantages of hdbscan include .











## Exercise: Apply kmeans to the moons dataset and compare the results. 
-- Be sure to try different numbers of centers.


## Exercise: Apply hdbscan to the moons dataset with different minimums for the number of points. 

## Exercise: Apply both kmeans and hdbscan to the `ChickWeight` dataset's "weight" "Time" variables, and see how well you can get each to perform.











## [US Census Data](https://elitedatascience.com/datasets)

- [US Census Data (Clustering)](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29) – Clustering based on demographics is a tried and true way to perform market research and segmentation.



## Links

- [Using clusterlab to benchmark clustering algorithms](https://www.r-bloggers.com/using-clusterlab-to-benchmark-clustering-algorithms/)
