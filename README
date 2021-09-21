# Documentation for the solution in Track 3 (DSP, UManitoba)

The submitted solution contains  two methods using Naive Bayes and Random Forest algorithms. There is another solution utilizing Convolutional Neural Net (CNN) that is not submitted as it has a longer training time. 

## Execution Details
There are *three* python files for Party 1, Party 2 and testing the model: **FederatedMLParty1.py**, **FederatedMLParty2.py** and **TestModel.py**. All three programs need configuration file: **idashTrack3.config** which has the following important parameters:

1. NetworkIDs
- Party1IP: IP address for Party 1 if we are doing a real-world communication scenario (set 127.0.0.1 otherwise)
- Party1Port: port number to connect (change port is found 'address in use error')
2. DataInfo
- PartyXNormal: Party1 or 2's normal dataset
- PartyXTumor: Party1 or 2's tumor dataset
- TestNormal/Tumor: The test dataset which is only accessed from **TestModel.py**
- CSVDelimiter: the delimiter for the dataset (, or \t)
3. TrainingInfo
- Algorithms: Please check with **NAIVEBAYES** and **RFOREST** (**CNN** not added)
- ReduceDimension: The number of Genes (or dimension) from the data to consider, increasing it will impact privacy and accuracy (If its set to 0 then it will consider the all values)
4. PrivacyParams
- Epsilon: The full (absolute maxium) privacy budget allowed for each party (i.e., if its set to 5 them P1 and P2 both will get a budget of 5 for all operations)
- CustomHistogram: If set yes, it creates a noisy histogram of the input dataset (details in DP Mechanism)
- ExponentialHistogram: If set yes, it creates a noisy histogram without any communication of the input dataset (details in DP Mechanism)
- NoisyMechanism: If set yes, it adds laplacian noise (using epsilon) to the inbetween shared statistics (only used in Naive Bayes)

No need to change the other parameters and they were for debugging purpose. All strings are without any quotations. If you get into  *Address in use* issue then  use (port 12124):
> sudo lsof -t -i tcp:12124 | xargs kill -9

### Training
Run `python3 FederatedMLParty1.py` which will create a server and listen for connection on *Party1Port* (defined on the config).  `python3 FederatedMLParty2.py`  will connect to this port and ip and start the privacy-preserving training process. The models will be saved in *saved_model* folder which will be used in testing. 
### Testing
To test the accuracy (precision, recall, auc are added) after the training execution from both parties, run `python3 TestModel.py`. It will test the model under current settings as defined in the config. The parameters TestNormal and TestTumor in the config denotes the test files.

### Differentially Private Mechanism
The differential privacy on both Naive Bayes and Random Forest methods are based on  ε-DP definition. The proposed solution guarantees the privacy of the input data going into the federated learning process. In other words, this solution is focused on data privacy compared to sharing private statistics in the federated learning process. 
Initially, both parties select a fixed number of genes to reduce the number of genes based on the variance. They only share the indices (no data) after sorting their individual gene-wise variances and agree on the fixed number of specific columns (change with `ReducedDimension` in config). Only these high-variance genes are selected for future computations.

The  privacy of each party's individual data is handled differently in `CustomHistogram` and `ExponentialHistogram` settings. Firstly, the value of the privacy budget, ε is equally split for all gene or columns.  In `CustomHistogram`, the parties share the noisy minimum and maximum,  based  on the `Report Noisy Max` mechanism. Then, they construct a histogram for each gene according to these min-max values and create a fixed number of bins (`Numbins` parameter) to discretize the data. When discretizing, I used the *Exponential mechanism* to select the correspoding bin for each data point. The utility function is based on the l1-distance between the original bin and each bin value. 
Due to the low variance in each gene, the other setting `ExponentialHistogram` does not share the noisy minimum or maximum values. Instead, it directly proceeds to the discretization of the gene values as each party converts their data without sharing any statistics. Since the final outputs from `CustomHistogram` and `ExponentialHistogram`  are both ε-differentially private, the computations done utilzing them should be private as well. The mechansims can be converted to  (ε,δ)  definition in future. 

Therefore, I implemented federated Naive Bayes, Random Forest and CNN that takes inputs from this smaller (reduced dimension) and private histogram version of the original dataset. The CNN implementation requires [PyGrid](https://github.com/OpenMined/PyGrid) which is a little complicated to install but has the real-world communication module for federated learning using PyTorch framework. Notably, there is another setting that allows privacy on the federated operations which  is defined in `NoisyMechanism` parameter (under development).

For any queries please email azizmma@cs.umanitoba.ca