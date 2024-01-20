import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets, utils
from utils import showimg, seed_everything, load_victims, load_surrogates, applyAS, load_local
import attacks.torchattacks as torchattacks
from pprint import pprint



def main(args):
    seed_everything(729)
    
    featuremap = []
    def hook_featuremap(module, input, output):
        featuremap.append(output)
    
    data_transform = transforms.Compose([transforms.CenterCrop(64), transforms.Grayscale(1), transforms.ToTensor()])
    sys_dataset = datasets.ImageFolder(root = './dataset/SAMPLE/synth', transform=data_transform)
    real_dataset = datasets.ImageFolder(root = './dataset/SAMPLE/real', transform=data_transform)
    sys_loader = torch.utils.data.DataLoader(sys_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last = False)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last = False)
    sys_num, real_num = len(sys_dataset), len(real_dataset)

    victim_model_names = ['aconv', 'shfullenet', 'mobilenet', 'regnet', 'efficientnet', 'densenet', 'resnet18', 'swint', 'convnext', 'vit', 'vgg16']
    victim_models = [load_victims(x) for x in victim_model_names]

    eps = args.epsilon
    steps = 10

    print('Attack: {}, Surrogate Model: {}, AS: {}'.format(args.attack, args.surrogate, args.AS))
    ASR = {m: 0. for m in victim_model_names}

    if args.local:
        surrogate = load_local(args.surrogate)
    else:
        surrogate = load_surrogates(args.surrogate)

    if args.AS:
        surrogate = applyAS(args.surrogate, surrogate, args.beta, args.decay)

    if args.attack == 'Mixup':
        mixupimg = torch.load('mixupimg.pt').cuda()
        if 'RN18' in args.surrogate:
            surrogate.maxpool.register_forward_hook(hook_featuremap)
            mixup_beta = 1.
        elif 'RN34' in args.surrogate:
            surrogate.maxpool.register_forward_hook(hook_featuremap)
            mixup_beta = 10.
        elif 'CNX' in args.surrogate:
            surrogate.features[0].register_forward_hook(hook_featuremap)
            mixup_beta = 1.

    # directly utilized from torchattacks at https://github.com/Harry24k/adversarial-attacks-pytorch, SVA is implemented into TIFGSM
    if args.attack == 'NI':
        attack = torchattacks.NIFGSM(surrogate, eps=eps, alpha=eps/8, steps=steps, decay=1.)
    elif args.attack == 'PGD':
        attack = torchattacks.PGD(surrogate, eps=eps, alpha=eps/8, steps=steps)
    elif args.attack == 'MI':
        attack = torchattacks.MIFGSM(surrogate, eps=eps, alpha=eps/8, steps=steps, decay=1.)
    elif args.attack == 'VT':
        attack = torchattacks.VMIFGSM(surrogate, eps=eps, alpha=eps/8, steps=steps, decay=0.)
    elif args.attack == 'DI':
        attack = torchattacks.DIFGSM(surrogate, eps=eps, alpha=eps/8, steps=steps)
    elif args.attack == 'TI':
        attack = torchattacks.TIFGSM(surrogate, eps=eps, alpha=eps/8, steps=steps, decay=0., diversity_prob=0., len_kernel=13)
    elif args.attack == 'SVA':
        attack = torchattacks.TIFGSM(surrogate, eps=eps, alpha=eps/8, steps=steps, sva=True, decay=1., random_start=False, len_kernel=13)

    seed_everything(37767)
    for i, ((sys_batch, labels), (real_batch, real_labels)) in enumerate(zip(sys_loader, real_loader)):
        sys_batch, labels, real_batch, real_labels = sys_batch.cuda(), labels.cuda(), real_batch.cuda(), real_labels.cuda()

        #Mixup-Attack adapted from https://github.com/YonghaoXu/UAE-RS, paper: https://arxiv.org/abs/2202.07054
        if args.attack == 'Mixup':
            momentum = torch.zeros_like(sys_batch).cuda()
            sys_adv_img = sys_batch.clone().cuda()
            featuremap = []
            surrogate(mixupimg)
            mixup_feature = featuremap[0]
            for atk_iter in range(10):
                sys_adv_img.requires_grad = True
                pred_loss = 0
                mix_loss = 0
                for kk in range(5):
                    featuremap = []
                    pred = surrogate(sys_adv_img/(2**(kk)))           #scale augmentation
                    pred_loss += torch.nn.CrossEntropyLoss()(pred, labels)
                    mix_loss += -1 * torch.nn.KLDivLoss()(featuremap[0], mixup_feature)
                total = pred_loss * mixup_beta + mix_loss
                grad = torch.autograd.grad(total, sys_adv_img, retain_graph=False, create_graph=False)[0]
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                grad = grad + momentum * 1.
                momentum = grad
                sys_adv_img = sys_adv_img.detach() + eps/8 * grad.sign() #grad / torch.norm(grad, float('inf'))
                delta = torch.clamp(sys_adv_img - sys_batch, min=-eps, max=eps)
                sys_adv_img = (sys_batch + delta).clamp(0, 1).detach()
        else:
            sys_adv_img = attack(sys_batch, labels)

        pert = (sys_adv_img - sys_batch.clone()).sign() * eps    #Sign-Projection, paper: https://arxiv.org/abs/2110.07718

        adv_target_img = (real_batch + pert).clamp(0, 1)


        with torch.no_grad():
            for m in range(len(victim_models)):
                ASR[victim_model_names[m]] += (torch.sum(victim_models[m](adv_target_img).argmax(1) != real_labels).float().item() / sys_num * 100)

    avg = 0
    for m in range(len(victim_model_names)):
        avg += ASR[victim_model_names[m]]
        ASR[victim_model_names[m]] = round(ASR[victim_model_names[m]], 2)
    avg /= len(victim_model_names)
    pprint(ASR, sort_dicts=False)
    print('Average ASR: {}%'.format(round(avg, 2)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epsilon', type=float, default = 16/255, help='lp-norm perturbation budget')
    parser.add_argument('--surrogate', type=str, default = 'RN18', help='surrogate model to be evaluated')
    parser.add_argument('--batch_size', type=int, default = 256)
    parser.add_argument('--attack', type=str, default = 'PGD', help='PGD, MI, NI, VT, DI, TI, Mixup, SVA')
    parser.add_argument('--beta', type=float, default = -1., help='beta for softplus')
    parser.add_argument('--decay', type=float, default = 1., help='decay for skip connections')
    parser.add_argument('--local', action='store_true', help='using local surrogates')
    parser.add_argument('--AS', action='store_true', help='using local surrogates')

    main(parser.parse_args())
