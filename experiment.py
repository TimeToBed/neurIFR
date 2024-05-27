
from utils import *
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torch import optim
from torch import nn
from models.types_ import *
import pytorch_lightning as pl
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics


class BSEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: nn.Module,
                 params,
                 text_logger = None) -> None:
        super().__init__()

        self.model = vae_model
        self.params = params
        self.indices_distrib = []
        self.curr_device = None
        self.highest_acc = 0

        self.text_logger = text_logger

    def get_rotation_task(self,imgs,labels):

        values = [0,90,180,270]
        for i in range(len(labels)):
            imgs[i] = F.rotate(imgs[i],values[labels[i]],InterpolationMode.BILINEAR)
        return imgs
            

    def forward(self, input, **kwargs) -> Tensor:
        outputs = {}
        keys = ['cla_vector','f_vector','ct_vector']
        if isinstance(input, list):
            if self.params['rotation'] and self.training:
                model_output = []
                r_labels = torch.randint(4,size=(len(input),len(input[0])),device=self.curr_device)
                for imgs,r_label in zip(input,r_labels):
                    imgs = self.get_rotation_task(imgs, r_label)
                    model_output.append(self.model(imgs).values())
                outputs.update({'r_labels':r_labels.reshape(-1)}) # num_crops,B -> num_crops*B 
            else:
                model_output = [self.model(imgs).values() for imgs in input]
            model_values = [torch.cat(list(vector)) for vector in zip(*model_output)]
            model_output = {k:v for k,v in zip(keys, model_values)}
            outputs.update(model_output)
        else:
            if self.params['rotation'] and self.training:
                r_labels = torch.randint(4,size=(len(input[0],)),device=self.curr_device)
                imgs = self.get_rotation_task(input, r_labels)
                outputs.update({'r_labels':r_labels})
                model_output = self.model(input)
            else:
                model_output = self.model(input)
            outputs.update(model_output)
        return outputs

    def test_step(self, batch, batch_idx, dataloader_idx):
        real_img, target = batch

        if dataloader_idx == 0:

            outputs = self.forward(real_img)
            acc = self.model.inference(outputs, target, 5, 5, is_MRR=self.params['is_MRR'])

        elif dataloader_idx == 1:

            outputs = self.forward(real_img)
            acc = self.model.inference(outputs, target, 5, 1, is_MRR=self.params['is_MRR'])

        #self.log('acc', acc)
        return acc
    
    def test_epoch_end(self, outputs):

        # 将所有批次的准确率求平均
        w5s5_acc, s5_interval = get_score(outputs[0])
        w5s1_acc, s1_interval = get_score(outputs[1])

        if self.text_logger is not None:

            self.text_logger.info('------------------------')
            self.text_logger.info('===BEST model test===')
            # 使用 self.log 记录 avg_acc，这将在每个 epoch 结束时自动记录
            self.text_logger.info('val_%d-way-%d-shot_acc:%.3f  interval:±%.2f'%(5,5,w5s5_acc*100,s5_interval*100))
            self.text_logger.info('val_%d-way-%d-shot_acc:%.3f  interval:±%.2f'%(5,1,w5s1_acc*100,s1_interval*100))


        print(f'5 way 5 shot: {(w5s5_acc*100):.2f}  interval:±{(s5_interval*100):.2f}')
        print(f'5 way 1 shot: {(w5s1_acc*100):.2f}  interval:±{(s1_interval*100):.2f}')
        return {"5w5s": w5s5_acc, "5w1s:": w5s1_acc}


    def training_step(self, batch, batch_idx):

        real_img, labels = batch
        self.curr_device = labels.device

        outputs = self.forward(real_img)

        #output = outputs['f_vector'] if self.params['mode'] == 'proto' else outputs['cla_vector']

        train_loss, acc = self.model.loss_function(outputs,
                                            labels, 
                                            params = self.params)

        return {'loss':train_loss, 'train_acc':acc}
    
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        
        train_acc = get_key_avg(outputs, key='train_acc')
        loss = get_key_avg(outputs, key='loss')

        self.text_logger.info("")
        self.text_logger.info("epoch %d/%d:" % (self.current_epoch+1,self.params['max_epochs']))
        self.text_logger.info("train_acc: %.3f" % (train_acc))


        scales = self.model.scale.detach().cpu().numpy()
        temperature = self.model.temperature.detach().cpu()
        for i in range(len(scales)):
            self.log(f'scale_{i}', scales[i])

        self.log('loss', loss)
        self.log('temperature', temperature)
        self.log('train_acc', train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx):

        real_img, target = batch

        outputs = self.forward(real_img)
        if dataloader_idx == 0:
            acc = self.model.inference(outputs, target, 5, 5)

        elif dataloader_idx == 1:
            acc = self.model.inference(outputs, target, 5, 1)
        elif dataloader_idx == 2:
            acc = self.model.inference(outputs, target, 5, 5)
        

        #self.log('acc', acc)
        return acc
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:

        # 将所有批次的准确率求平均
        w5s5_acc, w5s1_acc = sum(outputs[0])/len(outputs[0]),sum(outputs[1])/len(outputs[1])
        new_w5s5_acc = sum(outputs[2])/len(outputs[2])
        #avg_acc = sum(outputs)/len(outputs)

        self.text_logger.info("")
        self.text_logger.info("epoch %d/%d:" % (self.current_epoch+1,self.params['max_epochs']))
        self.text_logger.info('val_%d-way-%d-shot_acc:%.3f'%(5,5,w5s5_acc))
        self.text_logger.info('val_%d-way-%d-shot_acc:%.3f'%(5,1,w5s1_acc))
        self.text_logger.info('new_val_%d-way-%d-shot_acc:%.3f'%(5,5,new_w5s5_acc))
        
        #if self.current_epoch < 1: return
        
        if w5s5_acc >= self.highest_acc:
            self.highest_acc = w5s5_acc
            self.text_logger.info('BEST')
        # 使用 self.log 记录 avg_acc，这将在每个 epoch 结束时自动记录
        self.log('5w_5s_acc', w5s5_acc, prog_bar=True)
        self.log('5w_1s_acc', w5s1_acc, prog_bar=True)
        self.log('new_5w_5s_acc', new_w5s5_acc, prog_bar=False)


    def warm_up_cosine_lr_scheduler(self,optimizer, epochs, warm_up_epochs=5, min_scale=1e-2):
        """
        Description:
            - Warm up cosin learning rate scheduler, first epoch lr is too small
        Arguments:
            - optimizer: input optimizer for the training
            - epochs: int, total epochs for your training, default is 100. NOTE: you should pass correct epochs for your training
            - warm_up_epochs: int, default is 5, which mean the lr will be warm up for 5 epochs. if warm_up_epochs=0, means no need
            to warn up, will be as cosine lr scheduler
            - min_scale: float, setup ConsinAnnealingLR scale eta_min while warm_up_epochs = 0
        Returns:
            - scheduler
        """
        def lr_foo(epoch):      
            
            if epoch < warm_up_epochs :
                lr_scale = min_scale + (epoch / warm_up_epochs) * (1 - min_scale)
            else :
                lr_scale = 0.5 * (np.cos((epoch - warm_up_epochs) / (epochs - warm_up_epochs) * np.pi) + 1) * (1 - min_scale) + min_scale
            
            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        
        return scheduler
    
    
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.SGD(self.model.parameters(),
                               lr=self.params['LR'],
                               momentum= self.params['momentum'],
                               weight_decay=self.params['weight_decay'],
                               nesterov=self.params['nesterov'])
        """ optimizer = optim.Adam(self.model.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay']) """
        optims.append(optimizer)

        if self.params['scheduler'] == "MulStep":
            scheduler = optim.lr_scheduler.MultiStepLR(optims[0],
                                                   milestones=self.params['milestone'],
                                                   gamma=self.params['scheduler_gamma'])
        elif self.params['scheduler'] == "cosine":
            scheduler = self.warm_up_cosine_lr_scheduler(optims[0],
                                                        epochs=self.params['max_epochs'],
                                                        warm_up_epochs=0,
                                                        min_scale=self.params['min_scale'])
        scheds.append(scheduler)
        
        return optims, scheds
  

