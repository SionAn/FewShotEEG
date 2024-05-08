# SA-RelationNet & DA-RelationNet
The implementations of the two papers are as follows:
* ["Few-Shot Relation Learning with Attention for EEG-based Motor Imagery Classification, IROS 2020"](https://ieeexplore.ieee.org/abstract/document/9340933) <br> 
* ["Dual attention relation network with fine-tuning for few-shot EEG motor imagery classification, TNNLS"](https://ieeexplore.ieee.org/abstract/document/10167679) <br>

## Run
* Examples of 20-shot DA-RelationNet with EEGNet on BCI IV-2b dataset 
```
python main.py -a '_dual_att' -m 'EEGNet' -f 20 -tr '1,2,3,4,5,6,7,8' -te '9'
```
To test the trained model, use -t 'test'

## Citation
```bibtex
@inproceedings{an2020few,
  title={Few-shot relation learning with attention for EEG-based motor imagery classification},
  author={An, Sion and Kim, Soopil and Chikontwe, Philip and Park, Sang Hyun},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={10933--10938},
  year={2020},
  organization={IEEE}
}

@article{an2023dual,
  title={Dual attention relation network with fine-tuning for few-shot EEG motor imagery classification},
  author={An, Sion and Kim, Soopil and Chikontwe, Philip and Park, Sang Hyun},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
