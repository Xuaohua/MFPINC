# MFPINC V1.1
**MFPINC - Open-source ncRNA identification tool** 

Non-coding RNAs (ncRNAs) are recognized as pivotal players in the regulation of essential physiological processes such as nutrient homeostasis, development, and stress responses in plants. Common methods for predicting ncRNAs are susceptible to significant effects of experimental conditions and computational methods, resulting in the need for significant investment of time and resources. Therefore, in this paper, based on the PINC tool for identifying ncRNAs proposed by our lab in 2022, a new model for predicting identified ncRNAs, MFPINC, was proposed, and sequence features were carefully refined using variance thresholding and F-test methods, while deep features were extracted and feature fusion was performed by applying the GRU model. The comprehensive evaluation of multiple standard datasets shows that MFPINC not only achieves more comprehensive and accurate identification of gene sequences, but also significantly improves the expressive and generalization performance of the model, and MFPINC significantly outperforms the existing competing methods in ncRNA identification.  

MFPINC is an open source tool, and we welcome supplements and suggestions from all sectors of society.

This is the basic version of the tool, and some errors may occur. I hope the user can inform me of any errors they may encounter, or provide a solution that can be added to the MFPINC basic code.

MFPINC is suitable for Windows and Linux, but since some feature extraction needs to be done in the Linux environment, we recommend users to use the Linux system for extracting some features in the tool. The following are instructions for installing MFPINC.</br>

# AUTHOR/SUPPORT
* Zhenjun Nie - niezhenjun@stu.ahau.edu.cn;</br>
* Mengqing Gao - GaoMengqing@stu.ahau.edu.cn;</br>
* Corresponding author. Xiaodan Zhang - zxd@ahau.edu.cn

# Usage steps

## Installation

Before to start, it is necessary to install txCdsPredict.</br>

txCdsPredict:the precompiled binary is included in the package. Here is the link to download the source code:</br>

https://github.com/Zhenj-Nie/MFPINC/tree/master/kentUtils-master</br> 

All to install: https://github.com/TatianneNegri/RNAplonc/blob/master/Install.md

## txCdsPredict
* Please follow these steps before utilizing MFPINC:
1. `Installation of files`: Download the txCdsPredict folder in the CentOS7 system.
   
2. `Input of sequence`: Create a. fasta file and input the gene sequence you want to identify and identify, such as：
   >lcl|Carietinum_Ca_00004   
ATGATAACCACCTGGGTTTCGCAGCTGGAGCTTGAGCAGAAAGCCAAAACTCTGCATGAAGATATAATAAAGCATTGGATCTCAAGGGAATTAGCATTGTTACGAAATCACATTGATCTGGCAGATGAAAAAGGATGGAGTAGAGAATATCCTTTCATGATGTTTTTGTTGATGCTACCTCAGATTAGAAAGCTAAAAGGCCAAAATATTGAGAATTGTTAA
  
3. `txCdsPredict`: The txCdsPredict program is presented in the download sectionYour installation is explained in the install section.

    Command to execute: cd path/kentUtils/src/hg/txCds/txCdsPredict/
   
    The next command to execute: ./txCdsPredict result.fasta result.cds
   
    result.fasta = Name of the gene sequence from step 2
   
    result.cds = Output file name

**Once you have obtained the cds feature files using the txCdsPredict tool, our model will help further determine whether they are ncRNAs.**  

## MFPINC

### Required Packages
* Python 3.9.0 (or a compatible version) 
* Pytorch 1.13.0 (or a compatible version) 
* NumPy 1.23.3 (or a compatible version) 
* Pandas 1.2.4 (or a compatible version) 
* Scikit-learn 1.1.1 (or a compatible version)

### General Instructions for Conducting the CircPCBL Tool 

1. Prepare datasets (fasta file) 

2. Invoke CircPCBL with CMD 

3. Done! 
