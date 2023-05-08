***About***

This repository is a modification of Marc Finzi's [LieConv repository](https://github.com/mfinzi/LieConv#readme) that enables saving trained HLieResNet models and visualizing their predictions on a spring coupled mass system with equivariance in various Lie groups.

***Setup***

Python 3.9 is recommended. Install the original repository requirements with

        pip install -e .
        
***User guide***

[amolson2_jpalafou.model_config](https://github.com/MAE-DLPS/final-project-final_amolson2_jpalafou/blob/main/amolson2_jpalafou/model_config.py) specifies a simple set of model and training parameters. Train a simple fully connected model using
        
        python amolson2_jpalafou/train_springs_FC.py

Train HLieResNet models on spring coupled mass data with a specific LieConv group using

        python amolson2_jpalafou/train_springs_HLieResNet.py --group 'Trivial'
        python amolson2_jpalafou/train_springs_HLieResNet.py --group 'T2'
        python amolson2_jpalafou/train_springs_HLieResNet.py --group 'SO2'
        
If there is no data in [datasets](https://github.com/MAE-DLPS/final-project-final_amolson2_jpalafou/tree/main/datasets), it will be automatically generated. Each model is automatically saved in [models](https://github.com/MAE-DLPS/final-project-final_amolson2_jpalafou/tree/main/models).

Run

        python amolson2_jpalafou/run.py

the visualize the trajectories and momentums predicted by the four models. Note the difference in results when **center** is set to True or False for the HLieResNet models.
