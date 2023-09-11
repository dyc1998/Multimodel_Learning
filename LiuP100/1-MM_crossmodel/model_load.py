from torch import nn
import torch
import torch.nn.init as init
from Model import multimodalcnn

def init_feature_extractor(model, path):
    if path == 'None' or path is None:
        return
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    pre_trained_dict = checkpoint['state_dict']
    pre_trained_dict = {key.replace("module.", ""): value for key, value in pre_trained_dict.items()}
    print('Initializing efficient net')
    model.load_state_dict(pre_trained_dict, strict=False)
    #将原来的7类变成2类
    last_layer = list(model.children())[-1]
    last_layer = torch.nn.Linear(in_features=last_layer.in_features, out_features=2)
    model.fc = last_layer
    #是否进行权重的冻结？



def generate_model(opts):
    if opts.model == 'audio':
        print('Using {} model'.format(opts.model))
        model = multimodalcnn.AudioCNNPool(num_classes=2)
    elif opts.model=='visual':
        print('Using {} model'.format(opts.model))
        model = multimodalcnn.EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024],
                                                         im_per_sample=18, num_classes=2)
    elif opts.model=='audiovisual':
        print('Using {} model'.format(opts.model))
        model = multimodalcnn.MultiModalCNN(device=opts.device,num_classes=2, fusion='cs-lt')



    if opts.device != 'cpu':
        model = model.to(opts.device)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                                   p.requires_grad)
        print("{} model--->Total number of trainable parameters:{} ".format(opts.model,pytorch_total_params))

    return model