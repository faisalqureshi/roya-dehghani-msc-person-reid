# Person_ReIdentification

The original model:
First of all, plaese install all packages(Python==3.7.5, Pytorch==1.3.1, Cuda==9.2). To test the model, please run one of the scripts in script folder based on your dataset. 
the main file is iids_tnorm_self_kd.py in example folder. 
to install the required packages, you can use the following lines:
after creating an env in conda,
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch
just you need to install numpy and other main packages in requienment.txt, not the whole packages. 


Resnet after removing other techniques:
In stage1-stage2 branch, we have a reid model that i removed AIBN and TNorm techniques from backbone and run the model on the whole dataset. In order to train this model, you should run the train-market.sh in script folder in this branch. Plesea note that you should set your dataset path as the argument of --data-dir in thi train-market.sh. like this example:
python stage1-stage2.py      --data-dir    ../data/market1501       --dataset market1501 --checkpoint pretrained_weights/resnet50-0676ba61.pth

you can download the dataset from the following address:
https://drive.google.com/drive/folders/199jfb9a1gJy9ZUpAGIpEJI6vKGEcNnBo?usp=sharing

Branch fzq:
in this branch, just I did some experiments and the results are shown in results.txt in stage1-stage2 branch. 
