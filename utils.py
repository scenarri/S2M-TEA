import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from models.arch import AConvNets, shufflenetv2, mobilenetv2, regnet, efficientnet, densenet, resnet, swin_transformer, convnext, vision_transformer, vgg, convnext_tea, utils_iaa, convnext_iaa

def seed_everything(seed):

    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
        
def showimg(data, line=0):
    img = data.detach().cpu().numpy()[line][0]*255
    img = img.astype(np.uint8)
    map = cv2.applyColorMap(img, cv2.COLORMAP_PARULA)
    #map = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    plt.imshow(map[: , : , : : -1], vmin=0, vmax=1)
    plt.show()

def load_victims(model_name):
    if model_name == 'aconv':
        net = AConvNets.AConvNets64()
        net.load_state_dict(torch.load('./models/weights/victims/aconv.pth'))
    elif model_name == 'shfullenet':
        net = shufflenetv2.shufflenet_v2_x0_5()
        net.load_state_dict(torch.load('./models/weights/victims/shufflenetv2.pth'))
    elif model_name == 'mobilenet':
        net = mobilenetv2.mobilenet_v2()
        net.load_state_dict(torch.load('./models/weights/victims/mobilenetv2.pth'))
    elif model_name == 'regnet':
        net = regnet.regnet_y_400mf()
        net.load_state_dict(torch.load('./models/weights/victims/regnet.pth'))
    elif model_name == 'efficientnet':
        net = efficientnet.efficientnet_b0()
        net.load_state_dict(torch.load('./models/weights/victims/efficientnet.pth'))
    elif model_name == 'densenet':
        net = densenet.densenet121()
        net.load_state_dict(torch.load('./models/weights/victims/densenet.pth'))
    elif model_name == 'resnet18':
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./models/weights/victims/resnet18.pth'))
    elif model_name == 'swint':
        net = swin_transformer.swin_t()
        net.load_state_dict(torch.load('./models/weights/victims/swit.pth'))
    elif model_name == 'convnext':
        net = convnext.convnext_tiny()
        net.load_state_dict(torch.load('./models/weights/victims/convnext.pth'))
    elif model_name == 'vit':
        net = vision_transformer.vit_b_16()
        net.load_state_dict(torch.load('./models/weights/victims/vit.pth'))
    elif model_name == 'vgg16':
        net = vgg.vgg16()
        net.load_state_dict(torch.load('./models/weights/victims/vgg16.pth'))
    return net.eval().cuda()

def load_surrogates(model_name):
    if model_name == 'RN18':
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./models/weights/surrogates/RN18.pth'))
    elif model_name == 'RN18_FT':
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./models/weights/surrogates/RN18_FT.pth'))
    elif model_name == 'RN34':
        net = resnet.resnet34()
        net.load_state_dict(torch.load('./models/weights/surrogates/RN34.pth'))
    elif model_name == 'RN34_FT':
        net = resnet.resnet34()
        net.load_state_dict(torch.load('./models/weights/surrogates/RN34_FT.pth'))
    elif model_name == 'CNX':
        net = convnext_tea.convnext_tiny()
        net.load_state_dict(torch.load('./models/weights/surrogates/CNX.pth'))
    elif model_name == 'CNX_FT':
        net = convnext_tea.convnext_tiny()
        net.load_state_dict(torch.load('./models/weights/surrogates/CNX_FT.pth'))

    elif model_name == 'RN18_LRS':
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./models/weights/competitors/RN18_LRS.pth'))
    elif model_name == 'RN18_DRA':
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./models/weights/competitors/RN18_DRA.pth'))
    elif model_name == 'RN18_DSM':
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./models/weights/competitors/RN18_DSM.pth'))
    elif model_name == 'RN34_LRS':
        net = resnet.resnet34()
        net.load_state_dict(torch.load('./models/weights/competitors/RN34_LRS.pth'))
    elif model_name == 'RN34_DRA':
        net = resnet.resnet34()
        net.load_state_dict(torch.load('./models/weights/competitors/RN34_DRA.pth'))
    elif model_name == 'RN34_DSM':
        net = resnet.resnet34()
        net.load_state_dict(torch.load('./models/weights/competitors/RN34_DSM.pth'))
    elif model_name == 'CNX_LRS':
        net = convnext_tea.convnext_tiny()
        net.load_state_dict(torch.load('./models/weights/competitors/CNX_LRS.pth'))
    elif model_name == 'CNX_DRA':
        net = convnext_tea.convnext_tiny()
        net.load_state_dict(torch.load('./models/weights/competitors/CNX_DRA.pth'))
    elif model_name == 'CNX_DSM':
        net = convnext_tea.convnext_tiny()
        net.load_state_dict(torch.load('./models/weights/competitors/CNX_DSM.pth'))

    return net.eval().cuda()

def load_local(model_name):
    if 'RN18' in model_name:
        net = resnet.resnet18()
        net.load_state_dict(torch.load('./FT_result/'+model_name))
    elif 'RN34' in model_name:
        net = resnet.resnet34()
        net.load_state_dict(torch.load('./FT_result/'+model_name))
    elif 'CNX' in model_name:
        net = convnext_tea.convnext_tiny()
        net.load_state_dict(torch.load('./FT_result/'+model_name))

    return net.eval().cuda()

#from TransferAttackEval https://github.com/ZhengyuZhao/TransferAttackEval, you can also find SGM and LinBP there
def applyAS(model_name, net, beta=-1., decay=1.):
    if 'RN18' in model_name:
        if beta == -1. and decay == 1.:
            surrogate_AS = utils_iaa.resnet18(beta_value=3.246, decays=[0.751, 0.751, 0.751, 0.751]).cuda()
        else:
            surrogate_AS = utils_iaa.resnet18(beta_value=beta, decays=[decay, decay, decay, decay]).cuda()
    elif 'RN34' in model_name:
        if beta == -1. and decay == 1.:
            surrogate_AS = utils_iaa.resnet34(beta_value=1.156, decays=[0.751, 0.751, 0.751, 0.751]).cuda()
        else:
            surrogate_AS = utils_iaa.resnet34(beta_value=beta, decays=[decay, decay, decay, decay]).cuda()
    elif 'CNX' in model_name:
        if beta == -1. and decay == 1.:
            surrogate_AS = convnext_iaa.convnext_tiny_iaa(beta_value=1.04, decays=[0.82, 0.82, 0.82, 0.82]).cuda()
        else:
            surrogate_AS = convnext_iaa.convnext_tiny_iaa(beta_value=beta, decays=[decay, decay, decay, decay]).cuda()
    pre_dict = net.state_dict()
    iaa_dict = surrogate_AS.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in iaa_dict.keys()}
    iaa_dict.update(state_dict)
    model_dict = surrogate_AS.load_state_dict(iaa_dict)
    return surrogate_AS.eval()

def to_named_params(results, search_space, idx):
    params = results.x_iters[idx]#results.x
    param_dict = {}
    params_list  =[(dimension.name, param) for dimension, param in zip(search_space, params)]
    for item in params_list:
        param_dict[item[0]] = round(item[1], 3)
    return(param_dict)
