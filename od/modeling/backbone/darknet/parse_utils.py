import torch
from od.utils.yolo_utils import convert2cpu
import numpy as np

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def load_param(file, param):
    param.data.copy_(torch.from_numpy(
        np.fromfile(file, dtype=np.float32, count=param.numel()).reshape(param.shape)
    ))

#---------------------------------------------------
#Experiments for Speed
# def load_conv(file, conv_model):
#     load_param(file, conv_model.bias)
#     load_param(file, conv_model.weight)
#
# def load_conv_bn(file, conv_model, bn_model):
#     load_param(file, bn_model.bias)
#     load_param(file, bn_model.weight)
#     load_param(file, bn_model.running_mean)
#     load_param(file, bn_model.running_var)
#     load_param(file, conv_model.weight)

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    #conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(torch.reshape(torch.from_numpy(buf[start:start + num_w]),
    (conv_model.weight.shape[0],conv_model.weight.shape[1],conv_model.weight.shape[2],conv_model.weight.shape[3])));
    start = start + num_w
    return start

def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    #conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(torch.reshape(torch.from_numpy(buf[start:start + num_w]),
    (conv_model.weight.shape[0],conv_model.weight.shape[1],conv_model.weight.shape[2],conv_model.weight.shape[3])));
    start = start + num_w
    return start

def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]));   start = start + num_w
    return start

def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)

if __name__ == '__main__':
    import sys
    blocks = parse_cfg('cfg/yolo.cfg')
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)
