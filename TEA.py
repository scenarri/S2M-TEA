import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from utils import seed_everything, load_surrogates, Mkdir, to_named_params, load_local, applyAS
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
import argparse
import numpy as np
from tqdm import tqdm, trange
from pprint import pprint
import os
import matplotlib.pyplot as plt

def FT(args):

    data_transform = transforms.Compose([transforms.CenterCrop(64), transforms.Grayscale(1), transforms.ToTensor()])
    sys_dataset = datasets.ImageFolder(root = './dataset/SAMPLE/synth', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(sys_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True) #discard the last 'one' image for stable training

    surrogate_ori = load_surrogates(args.surrogate).train()
    surrogate = load_surrogates(args.surrogate).eval()

    optimizer = optim.SGD(surrogate.parameters(), lr=(0.005 if 'RN' in args.surrogate else 0.001), momentum=0.95, weight_decay=0.0006)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = np.random.randint(1e4, 1e5)

    print('perform fine-tuing with L_FT = L_CE - lambda * L_Data with seed {}...'.format(seed))
    seed_everything(seed)

    for epoch in tqdm(range(20), desc='Fine-tuning'): #20 epochs
        surrogate.train()
        surrogate_ori.train()
        for i, (sys_data, sys_label) in enumerate(train_loader):
            sys_data, sys_label = sys_data.cuda(), sys_label.cuda()

            le = sys_data.size(0)
            optimizer.zero_grad()

            sys_data.requires_grad = True
            new_sys = torch.autograd.grad(nn.CrossEntropyLoss(reduction='mean')(surrogate(sys_data), sys_label), sys_data, retain_graph=True, create_graph=True)[0]

            Ldata = 0
            for num in range(args.samples):
                sub_data = (sys_data + args.sigmaFT * torch.randn_like(sys_data)).clamp(0, 1).detach().clone()
                sub_data.requires_grad = True
                new_sub1 = torch.autograd.grad(nn.CrossEntropyLoss(reduction='mean')(surrogate(sub_data), sys_label), sub_data, retain_graph=True, create_graph=True)[0]
                Ldata += (torch.cosine_similarity(new_sys.reshape(le, -1), new_sub1.reshape(le, -1))).mean()

            # L_Data
            loss = -1 * args.lambda_ * Ldata/args.samples + 1 * nn.CrossEntropyLoss(reduction='mean')(surrogate(sys_data.clone()), sys_label)

            loss.backward()
            optimizer.step()
        scheduler.step()
        surrogate.eval()
        surrogate_ori.eval()

    Mkdir('./FT_result')
    name = args.surrogate + '_seed-' + str(seed) + '_FT.pth'
    torch.save(surrogate.state_dict(), './FT_result/' + name)
    print('weight has been saved as ' + './FT_result/' + name)

    return name

def AS(args, name):

    data_transform = transforms.Compose([transforms.CenterCrop(64), transforms.Grayscale(1), transforms.ToTensor()])
    sys_dataset = datasets.ImageFolder(root='./dataset/SAMPLE/synth', transform=data_transform)
    sys_loader = torch.utils.data.DataLoader(sys_dataset, batch_size=128, shuffle=False, num_workers=0)

    space = [Real(0.001, 10, name='beta'),
             Real(0.05, 1, name='decay')]

    @use_named_args(space)
    def objective(beta, decay):
        surrogate_ori = load_surrogates(args.surrogate.replace('_FT', '')).eval()
        surrogate = load_local(name).eval()
        surrogate = applyAS(args.surrogate, surrogate, beta, decay)

        loss = []
        for epoch in range(1):
            for i, (sys_data, sys_label) in enumerate(sys_loader):
                le = sys_data.size(0)
                sys_data.requires_grad = True
                sys_data, sys_label, = sys_data.cuda(), sys_label.cuda()
                new_sys = torch.autograd.grad(nn.CrossEntropyLoss(reduction='sum')(surrogate(sys_data), sys_label), sys_data, retain_graph=False)[0]

                sys_data = sys_data.detach().clone()
                sys_data.requires_grad = True
                ori_sys = torch.autograd.grad(nn.CrossEntropyLoss(reduction='sum')(surrogate_ori(sys_data), sys_label), sys_data, retain_graph=False)[0]

                sub_data = (sys_data + args.sigmaAS * torch.randn_like(sys_data)).clamp(0, 1).detach().clone()
                sub_data.requires_grad = True
                new_sub = torch.autograd.grad(nn.CrossEntropyLoss(reduction='sum')(surrogate(sub_data), sys_label), sub_data, retain_graph=False)[0]

                #L_Total = 1/2(L_Data + L_Model)
                loss.append(
                    (0.5 * (torch.cosine_similarity(new_sub.reshape(le, -1), new_sys.reshape(le, -1)))
                     + 0.5 * (torch.cosine_similarity(new_sys.reshape(le, -1), ori_sys.reshape(le, -1)))).mean().item())
        return -1 * np.mean(loss)

    print('AS: searching architecture hyper-parameters for higher transferability estimation L_Total = L_Data + L_Model...')
    results = gp_minimize(objective, space, n_calls=50, random_state=None, n_random_starts=10, verbose=args.verbose, kappa=1.96, xi=0.01)
    ord = results.func_vals.argsort()
    top5 = [to_named_params(results,space,idx=ord[j]) for j in range(5)]
    print('The top 5 params are')
    pprint(top5)
    #skopt.dump(results, 'bayes_opt_results.pkl')
    #skopt.plots.plot_convergence(results)
    #plt.show()


    FT_cmd = 'python transfer_eval.py --local --surrogate ' + name
    FT_AS_cmd = 'python transfer_eval.py --local --AS --surrogate ' + name + ' --beta ' + str(top5[0]['beta']) + ' --decay ' + str(top5[0]['decay'])

    return FT_cmd, FT_AS_cmd




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--surrogate', type = str, default = 'RN18', help='surrogate model to be fine-tuned')
    parser.add_argument('--sigmaFT', type = float, default = 0.2, help='\sigma_FT')
    parser.add_argument('--lambda_', type = float, default = 1., help='\lambda')
    parser.add_argument('--samples', type = int, default = 5, help='number of Gaussian neighbor sampled')
    parser.add_argument('--seed', type = int, default = None, help='favorite random state')

    parser.add_argument('--sigmaAS', type = float, default = 0.3, help='\sigma_AS')
    parser.add_argument('--name', type=str, default = None, help='existing FT weights')
    parser.add_argument('--test', action = 'store_true', help = 'using local surrogates')
    parser.add_argument('--verbose', action = 'store_true', help = 'control the verbosity')

    args = parser.parse_args()

    if args.name is None:
        name = FT(args)
        FT_cmd, FT_AS_cmd = AS(args, name)
    else:
        FT_cmd, FT_AS_cmd = AS(args, args.name)

    print()
    print('testing FT-surrogate with the following command...')
    print(FT_cmd)
    if args.test:
        os.system(FT_cmd)

    print()
    print('testing FT+AS surrogate with the following command...')
    print(FT_AS_cmd)
    if args.test:
        os.system(FT_AS_cmd)
