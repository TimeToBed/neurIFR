import torch.nn as nn
from .layers import *
from torchvision.models.resnet import conv1x1,conv3x3
from typing import Any, Callable, List, Optional
from torchvision.models import ResNet50_Weights,ResNet18_Weights
import torch.nn.functional as F

def pdist_l2(x,y):
    '''
    Input: x is a Nxd matrix
        y is an optional Mxd matirx
    Output: dist_metrix[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist_metrix = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist_metrix

def pdist_cos(x,y):
    '''
    Input: x is a Nxd matrix
        y is an optional Mxd matirx
    Output: dist[i,j] =  x * y /(||x|| * ||y||)
    '''
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    sim_matrix = torch.matmul(x, y.T)
    return sim_matrix

def sim_matrix_KL(embeddings:torch.Tensor,temperature):

    B,C = embeddings.shape
    p_log_prob = F.log_softmax(embeddings,dim=1)
    q_prob = F.softmax(embeddings,dim=1)
    p_log_prob_exp = p_log_prob.unsqueeze(1).expand(B, B, C) # B, 1*B, C
    q_prob_exp = q_prob.unsqueeze(0).expand(B, B, C) # 1*B, B, C

    # 直接在大的张量上应用 F.kl_div
    sim_matrix = F.kl_div(p_log_prob_exp, q_prob_exp, reduction='none').sum(2).neg()
    sim_matrix.fill_diagonal_(-1e7)
    sim_matrix = torch.exp(sim_matrix * temperature)

    return sim_matrix

def sim_matrix_cos(x,temperature):

    sim_matrix = pdist_cos(x,x)
    sim_matrix.fill_diagonal_(-1e7)
    sim_matrix = torch.exp(sim_matrix * temperature)
    return sim_matrix

def contrastive_loss(embeddings:torch.Tensor, labels, temperature):

    #可以更换相似度计算公式，也可以在分类所得的概率后进行kl散度，或者mse，cos相似度等操作

    # 计算相似度矩阵利用L2距离
    #sim_matrix = pdist_l2(embeddings, embeddings).neg()
    #sim_matrix.fill_diagonal_(-1e7)
    #sim_matrix = torch.exp(sim_matrix / embeddings.shape[-1]* temperature)

    #利用余弦相似度
    sim_matrix = sim_matrix_cos(embeddings,temperature)
    #sim_matrix = sim_matrix_KL(embeddings,temperature)

    mask_positive = labels.view(-1, 1) == labels.view(1, -1)  # 计算正样本对 Mask
    mask_positive.fill_diagonal_(0)  # 排除自身和自身的比较

    pos_sim = torch.stack([sim_matrix[i,s_pos].sum() for i,s_pos in enumerate(mask_positive)])
    all_sim = sim_matrix.sum(dim=1)
    pred = pos_sim/(all_sim + 1e-8)
    pred = torch.where(pred == 0, torch.ones_like(pred), pred)
    loss = torch.mean(-torch.log(pred) * 0.5)

    return loss

def sample_contrastive_loss(embeddings:torch.Tensor, num_crops, temperature=1):

    batch_size = embeddings.shape[0]//num_crops

    # 计算相似度矩阵（可以采用余弦相似度，点积等方法）
    #sim_matrix = pdist(embeddings, embeddings).neg()
    #sim_matrix.fill_diagonal_(-1e7)
    #sim_matrix = torch.exp(sim_matrix / embeddings.shape[-1]* temperature)[:batch_size]

    #利用余弦相似度
    sim_matrix = sim_matrix_cos(embeddings,temperature)[:batch_size]

    mask_positive = torch.cat([torch.zeros(batch_size,batch_size)]+[torch.eye(batch_size)]*(num_crops-1),dim=0).T.bool()

    pos_sim = torch.stack([sim_matrix[i,s_pos].sum() for i,s_pos in enumerate(mask_positive)])
    all_sim = sim_matrix.sum(dim=1)
    pred = pos_sim/all_sim
    loss = torch.mean(-torch.log(pred))

    return loss



class FResNet(nn.Module):

    def __init__(self, block, n_blocks, drop_rate=0.0, drop_block=False):
        super().__init__()

        self.inplanes = 3
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=5)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=5)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

 
    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,max_pool=max_pool)
            #layer = block(self.inplanes, planes, stride, downsample)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
            #layer = block(self.inplanes, planes, stride, downsample)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward_feature(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

    def forward(self, x, is_feat=False):

        x = self.forward_feature(x)[-1]

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class MetricNet(nn.Module):

    def __init__(self, channels:List, **kwargs) -> None:
        super().__init__()
        self.channels = channels
        """The model structure will be released after the paper is accepted."""
    
    def forward(self, x_list):

        m4 = self.metric4(x_list[3])

        return [m4]
    
def Fresnet12(drop_rate=0.0, dropblock=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = FResNet(FBasicBlock, [1, 1, 1, 1], drop_rate=drop_rate, drop_block=dropblock, **kwargs)

    return model

