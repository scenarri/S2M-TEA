import torch
from torchvision import transforms, datasets
from utils import load_victims
from pprint import pprint


def main():
    data_transform = transforms.Compose([transforms.CenterCrop(64), transforms.Grayscale(1), transforms.ToTensor()])
    real_dataset = datasets.ImageFolder(root='./dataset/SAMPLE/real', transform=data_transform)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
    real_num = len(real_dataset)

    victim_model_names = ['aconv', 'shfullenet', 'mobilenet', 'regnet', 'efficientnet', 'densenet', 'resnet18', 'swint',
                          'convnext', 'vit', 'vgg16']
    victim_models = [load_victims(x) for x in victim_model_names]
    ACC = {m: 0. for m in victim_model_names}

    for i, (real_batch, real_labels) in enumerate(real_loader):
        real_batch, real_labels = real_batch.cuda(), real_labels.cuda()

        with torch.no_grad():
            for m in range(len(victim_models)):
                ACC[victim_model_names[m]] += (torch.sum(victim_models[m](real_batch).argmax(1) == real_labels).float().item() / real_num * 100)


    for m in range(len(victim_model_names)):
        ACC[victim_model_names[m]] = round(ACC[victim_model_names[m]], 2)
    print('acc:')
    pprint(ACC, sort_dicts=False)


if __name__ == '__main__':

    main()
