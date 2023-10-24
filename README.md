The source code of the FAACL: Federated Adaptive Asymmetric Clustered Learning:


FAACL can simulate the following CFL frameworks:

**FedAvg** -> [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html)

**FedGroup** -> [FedGroup: Efficient Clustered Federated Learning via Decomposed Data-Driven Measure](https://arxiv.org/abs/2010.06870)

**IFCA** -> [An Efficient Framework for Clustered Federated Learning](https://proceedings.neurips.cc/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html)

**FeSEM** -> [Multi-center federated learning](https://arxiv.org/abs/2005.01026)

**FedDrift** -> [Federated Learning under Distributed Concept Drift](https://proceedings.mlr.press/v206/jothimurugesan23a.html)

Requried packages is in requirements.txt, to install them, run 

    pip install -r requirements.txt
 
 
Please download the flearn.trainer folder from FlexCFL repository and put them in flearn directory.

The flearn folder should look like the following: 

    FAACL -> flearn -> trainer -> fedavg.py

                               -> fedgroup.py

                               ...
                                       
                    

Please download the mnist, emnist and fmnist from [FedProx](https://github.com/litian96/FedProx/tree/master) repository, fmnist from [ditto](https://github.com/litian96/ditto/tree/master) repository, femnist from [LEAF](https://github.com/TalwalkarLab/leaf) repository

The data folder should look like the following:


    FAACL -> data -> mnist -> data -> train -> train.json
    
                                   -> test  -> test.json
                                   
                    fmnist -> data -> train -> train.json
                    
                                   -> test -> test.json

                    ...

To reproduce the Fedavg, FeSEM, FedGroup, and IFCA, run 

    python test-baseline.py

To reproduce the Centralize method, run 

    python test-centralize.py

To reproduce the FAACL method, run 

    python test-FAACL.py

To reproduce the FedDrift method, run 

    python test-FedDrift.py



All evaluation results will be stored in excel format files, placed in results/ directory.



