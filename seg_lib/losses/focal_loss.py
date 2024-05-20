"""
MIT License

Copyright (c) 2023 Xian Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.alpha = self.__set_alpha(alpha, num_classes)
        self.gamma = gamma
        self.num_classes = num_classes

    def __set_alpha(self, alpha, num_classes):
        if isinstance(alpha, list):
            if len(alpha) != num_classes:
                raise TypeError(
                    '"alpha" length do not match the number of classes: '
                    f'{len(alpha)} != {num_classes}'
                )
            print(
                f'Focal loss alpha={alpha}, '
                'will assign alpha values for each class'
            )
            return torch.Tensor(alpha)
        
        if alpha >= 1:
            raise TypeError('"alpha" should be smaller than 1.')
        print(
            f'Focal loss alpha={alpha}, '
            'will shrink the impact in background'
        )
        local_alpha = torch.zeros(num_classes)
        local_alpha[0] = alpha
        local_alpha[1:] = 1 - alpha

        return local_alpha

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds:
            size: [B, N, C] or [B, C], corresponds to detection and
                  classification tasks  [B, C, H, W]: segmentation
        :param labels:
            size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(
            torch.pow((1 - preds_softmax), self.gamma), preds_logsoft
        )

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
