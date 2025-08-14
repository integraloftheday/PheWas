using_version("Hmisc","4.8-0")
using_version("rms","6.4-1")
using_version("ggsci","3.0.0")
#using_version("miceadds","3.16-18")
#using_version("miceadds","3.17-44")
using_version("ggpubr","0.5.0")

load_stored_result <- FALSE

library(nanoparquet)

#load packages and dataset 

library(stringr)
library(bigrquery)
library(data.table)
library(readr)
library(Hmisc)
library(ggplot2)
library(ggrepel)
library(viridis)
library(ggsci)
#library(miceadds)
library(rms)
library(ggpubr)
library(tidyverse)
library(lubridate)
#library(arrow)
library(dplyr)
library(data.table)
library(purrr)