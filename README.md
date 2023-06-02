# Code_TP
Using a pathogenic microorganism DNA capture-based NGS technique for pathogenic microorganism DNA enrichment sequencing, followed by machine learning to establish pathogenic microorganism diagnostic thresholds that leaded to a pathogenic microorganism diagnostic model. 

## arrange the data and generate a sample information file, format like example/sample

## generate pipeline scripts for each step
python3 Meta/hisat2.py example/sample
## running the pipeline
bash kraken.sh
bash bracken.sh
##DecisionTree
python3 DecisionTree/Decision_Tree2.py example/data.tsv



This study was supported by the Natural Science Fund of China (82072328), Capital’s Funds for Health Improvement and Research (CFH2020-4-2163), Beijing Municipal Administration of Hospitals’ Ascent Plan (DFL20181602), Beijing Hospitals Authority Youth Programme (QML20201601), and Tongzhou “Yun He” Talent Project (YHLD2019001 and YHLD2018030).
