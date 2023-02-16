export PYTHONPATH=$PYTHONPATH:./
python iids_tnorm_self_kd.py --dataset market1501 --data-dir ../iids/example/data --evaluate --checkpoint ../iids/pretrained_weights/market.pth.tar