import torch
from torch.autograd import Variable
from utils.Loss import CEBayesRiskLoss, KLDivergenceLoss, SSBayesRiskLoss

def val(model, dataloader, epoch, args):
    model.eval()

    loss = 0.0
    bayes_risk = CEBayesRiskLoss().to(args.device)
    kld_loss = KLDivergenceLoss().to(args.device)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data  
            inputs, labels = Variable(inputs.float()).to(args.device), Variable(labels).to(
                args.device)
            evidences = model(inputs)
            eye = torch.eye(4, dtype=torch.float32, device=args.device)
            labels = eye[labels]
            annealing_coef = min(1.0, epoch / args.epochs)
            loss += bayes_risk(evidences, labels) + annealing_coef * kld_loss(evidences, labels)

        return loss / len(dataloader)