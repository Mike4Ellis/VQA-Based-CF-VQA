import torch
import torch.nn as nn
from block.models.networks.mlp import MLP
from .utils import grad_mul_const # mask_softmax, grad_reverse, grad_reverse_mask, 


class RUBiNet(nn.Module):
    """
    Wraps another model
    The original model must return a dictionnary containing the 'logits' key (predictions before softmax)
    Returns:
        - logits: the original predictions of the model
        - logits_q: the predictions from the question-only branch
        - logits_rubi: the updated predictions from the model by the mask.
    => Use `logits_rubi` and `logits_q` for the loss
    """
    '''
    - logits: 模型原预测结果
    - logits_q: question-only分支预测结果
    - logits_rubi: 模型mask后新预测结果
    '''
    def __init__(self, model, output_size, classif, end_classif=True):
        super().__init__()
        self.net = model
        self.c_1 = MLP(**classif)
        self.end_classif = end_classif
        if self.end_classif:
            self.c_2 = nn.Linear(output_size, output_size)

    def forward(self, batch):
        out = {}
        # 模型预测结果
        # 其中'logits'对应模型原预测结果
        # 'q_emb'对应问题编码
        net_out = self.net(batch)
        logits = net_out['logits']
        # 问题编码 对应论文e_q
        q_embedding = net_out['q_emb']  # N * q_emb
        # 阻止nn_q到e_q的反向传播
        # 自定义函数 使backward()时梯度*0 得梯度为0
        q_embedding = grad_mul_const(q_embedding, 0.0) # don't backpropagate through question encoder
        # MLP处理 对应论文nn_q
        q_pred = self.c_1(q_embedding)
        # mask操作
        fusion_pred = logits * torch.sigmoid(q_pred)
        
        # 是否添加最后的分类层 对应论文c_q 
        if self.end_classif:
            q_out = self.c_2(q_pred)
        else:
            q_out = q_pred

        out['logits'] = net_out['logits']
        out['logits_all'] = fusion_pred
        out['logits_q'] = q_out
        return out

    def process_answers(self, out, key=''):
        out = self.net.process_answers(out)
        out = self.net.process_answers(out, key='_all')
        out = self.net.process_answers(out, key='_q')
        return out
