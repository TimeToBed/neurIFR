import torch

from .mymodels import *
import random
import numpy as np
from sklearn import metrics


class Baseline(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 represent_dim: int,
                 num_classes: int,
                 **kwargs) -> None:
        super().__init__()
        self.represent_dim = represent_dim
        self.hidden_dim = hidden_dim
        """The model structure will be released after the paper is accepted."""


    def forward(self, x):

        features = self.extractor.forward_feature(x)

        outputs = self.metric(features)
        
        cla_vector = self.cla_head(outputs[0])
        f_vector = outputs[0]
        ct_vector = self.represent(cla_vector)

        return {'cla_vector':cla_vector, 'f_vector':f_vector,'ct_vector':ct_vector}
    
    def merge_crops(self, feature, way, shot, num_crops, mode):

        Bn, C = feature.shape

        if mode == 'finetune':
            feature = feature.reshape(num_crops, Bn//num_crops, C)
            feature = feature.permute(1,0,2).contiguous()
            x = feature.reshape(way,-1,C)
            indices = torch.randperm(x.size(1))
            x = x[:,indices]
            merge_x = x.view(-1,C)

        elif mode == 'test':
            
            feature = feature.reshape(num_crops, Bn//num_crops, C)
            feature = feature.permute(1,0,2).contiguous() # B, num_crops, C
            feature = feature.view(way, -1, num_crops, C) #way, s+q, num_crops, C
            x_s = torch.mean(feature[:,:shot,:,:],dim=2)
            x_s = torch.cat([x_s,feature[:,:shot,0,:]],dim=-1)
            x_q = torch.mean(feature[:,shot:,:,:],dim=2)
            x_q = torch.cat([x_q,feature[:,shot:,0,:]],dim=-1)
            #x_q = feature[:,shot:,0,:]
            x = torch.cat([x_s]+[x_q], dim=1)
            merge_x = x.view(-1,C*2)
        else:
            assert False, "mode not be defined"

        return merge_x

    
    def inference(self, outputs, target, way, shot, query=15, is_MRR=False):

        f_vector = outputs['f_vector']
        if is_MRR:
            f_vector = self.merge_crops(f_vector,way,shot,5,mode='test')
        neg_l2_dist,_ = self.get_neg_dist(f_vector, way, shot, query)
        output = self.get_logits(neg_l2_dist)
        pred = torch.argmax(output, dim=1, keepdim=False).cpu()
        target = torch.LongTensor([i//query for i in range(query*way)])

        acc = metrics.accuracy_score(target.cpu(), pred)
        return acc

    def get_neg_dist(self,feature_vector,way,shot,query_shot, times=0):
        #feature_vector: way*(shot+query),C
        feature_vector = feature_vector.reshape(way, shot+query_shot, feature_vector.shape[-1])
        support = feature_vector[:,:shot].contiguous().view(way,shot,-1)
        query = feature_vector[:,shot:].contiguous().view(way*query_shot,-1) # way*query_shot,dim

        if self.training and times != 0:
            centroid_list = []
            for _ in range(times):
                time = random.randint(1,shot)
                m_list = np.random.choice(list(range(shot)),time,replace=False).tolist()
                centroid_list.append(torch.mean(support[:,m_list,:],dim=1))
            centroid = sum(centroid_list)
            #s_q = torch.stack(centroid_list,dim=0).permute(1,0,2).contiguous().view(-1,centroid.shape[-1]) #w*t,d
            #neg_s_dist = pdist(s_q, centroid).neg().view(way,times,way)
        else:
            centroid = torch.mean(support,dim=1) # way, dim
            #neg_s_dist = None

        neg_l2_dist = pdist_l2(query,centroid).neg().view(way*query_shot,way)
        #neg_l2_dist = pdist_cos(query,centroid).view(way*query_shot,way)
        
        return neg_l2_dist, None

    def get_logits(self, neg_l2_dist):

        logits = neg_l2_dist/self.hidden_dim * self.scale
        #logits = neg_l2_dist * self.scale #cos相似度

        return logits



        
