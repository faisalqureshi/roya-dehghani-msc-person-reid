export PYTHONPATH=$PYTHONPATH:./
python iids_tnorm_self_kd.py      --data-dir  /home/rdehghani/intra-inter-resnet/Person_ReIdentification/v1-reid/data       --dataset market1501 --checkpoint pretrained_weights/market.pth.tar

#v2-reid/connext_without_Tnorm_AIBN/pretrained_weights/convnext_tiny_1k_224_ema.pth