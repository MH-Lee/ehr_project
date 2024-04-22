import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        """
        Arguments:
        embeddings -- torch.Tensor of shape (2 * batch_size, embed_dim),
                      where 'embeddings' are concatenated outputs of [h_i and h_i+]
                      for all i in the batch, thus 2 * batch_size in total.

        Returns:
        loss -- scalar tensor, the computed SimCSE unsupervised contrastive loss
        """
        batch_size = embeddings.size(0) // 2

        # Cosine similarity matrix calculation
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2) / self.temperature

        # Extracting the diagonal elements, which are the similarities between each h_i and its corresponding h_i+
        positives = sim_matrix.diag()[1::2]  # start from index 1 with steps of 2

        # Summation of exponentials of the cosine similarities for each row, excluding self-similarity
        sum_exp = sim_matrix.exp().sum(dim=1) + sim_matrix.exp().sum(dim=0) - sim_matrix.exp().diag()

        # Negative log likelihood loss computation for each positive pair
        loss = -torch.log(positives / sum_exp[::2])  # take every second element starting from 0

        return loss.mean()  # Mean loss over all positive pairs in the batch
    

# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(ContrastiveLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, embeddings_1, embeddings_2):
#         # 임베딩 정규화
#         embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
#         embeddings_2 = F.normalize(embeddings_2, p=2, dim=1)
#         # 코사인 유사도 계산
#         cosine_sim = torch.matmul(embeddings_1, embeddings_2.transpose(1,0)) / self.temperature
#         # 대각선(긍정적 쌍)에 대한 손실 계산
#         batch_size = embeddings_1.size(0)
#         labels = torch.arange(batch_size, device=embeddings_1.device)
#         loss = F.cross_entropy(cosine_sim, labels)
#         return loss

    
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids. data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(inputs.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss