#!/usr/bin/bash

# for i in {1..5};
# do
#     for j in {0..19};
#     do
#        python train.py --exp_name test/fold-${i}/epoch-${j} --ckpt_path ./exps/covidnet/${i}/models/covidnet_${j}.pth
#     done
# done
i=1
j=11
python train.py --exp_name test/fold-${i}/epoch-${j} --ckpt_path ./exps/covidnet/${i}/models/covidnet_${j}.pth
i=2
j=5
python train.py --exp_name test/fold-${i}/epoch-${j} --ckpt_path ./exps/covidnet/${i}/models/covidnet_${j}.pth
i=3
j=19
python train.py --exp_name test/fold-${i}/epoch-${j} --ckpt_path ./exps/covidnet/${i}/models/covidnet_${j}.pth
i=4
j=11
python train.py --exp_name test/fold-${i}/epoch-${j} --ckpt_path ./exps/covidnet/${i}/models/covidnet_${j}.pth
i=5
j=7
python train.py --exp_name test/fold-${i}/epoch-${j} --ckpt_path ./exps/covidnet/${i}/models/covidnet_${j}.pth

# python train.py --exp_name test/fold-1 --ckpt_path ./exps/covidnet/1/models/covidnet_12.pth
# python train.py --exp_name test/fold-2 --ckpt_path ./exps/covidnet/2/models/covidnet_9.pth
# python train.py --exp_name test/fold-3 --ckpt_path ./exps/covidnet/3/models/covidnet_14.pth
# python train.py --exp_name test/fold-4 --ckpt_path ./exps/covidnet/4/models/covidnet_7.pth
# python train.py --exp_name test/fold-5 --ckpt_path ./exps/covidnet/5/models/covidnet_11.pth