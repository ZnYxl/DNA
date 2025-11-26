import torch
from torch.autograd import Variable
from utils.utils import statistics_s

def test_s(model, dataloader, args):
    model.eval()
    model.to(args.device)
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs.float()).to(args.device), Variable(labels).to(
                args.device)
            evidences = model(inputs)
            alphas = evidences + 1.0
            strength = torch.sum(alphas, dim=-1, keepdim=True)
            probabilities = alphas / strength
            y = probabilities.argmax(dim=2)
            if i == 0:
                print("预测：", y[0])
                print("标签：", labels[0])
            correct += statistics_s(y, labels)
            total += labels.size(0)
        avg_accuracy = correct / total * 100

        return avg_accuracy





