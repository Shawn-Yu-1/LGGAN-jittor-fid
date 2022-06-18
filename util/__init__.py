"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor as jt


def preprocess_input(self, data):
    # move to GPU and change data types
    data['label'] = data['label'].long()
    # if self.use_gpu():
    #     data['label'] = data['label']
    #     data['instance'] = data['instance']
    #     data['image'] = data['image']

    # create one-hot label map
    label_map = data['label'] # print(label_map.size()) [1, 1, 256, 512]
    bs, _, h, w = label_map.size()
    nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
        else self.opt.label_nc
    input_label = jt.zeros(shape=(bs, nc, h, w)) # print(input_label.size()) [1, 35, 256, 512]
    input_semantics = input_label.scatter_(1, label_map, jt.array(1.0)) # print(input_semantics.size()) [1, 35, 256, 512]
    
    return input_semantics
