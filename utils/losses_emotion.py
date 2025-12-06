'''
losses_emotion.py: 감정 분류(emotion classification) 모델을 학습할 때 사용할 Loss Function들을 정의하고 선택하는 파일
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config_TomJerry import LOSS_NAME, LABEL_SMOOTHING, FOCAL_GAMMA


# ===========================================================
# FocalLoss Class
# 감정 데이터는 클래스 불균형이 심하기 때문에
# 쉬운 샘플 → loss 감소 (학습 거의 안 함)
# 어려운 샘플 → loss 증가 (더 많이 학습) 할 수 있도록 focal loss 도입
# ===========================================================
class FocalLoss(nn.Module):
    """
    multi-class focal loss
    class-level로 gamma, reduction 설정
    """
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):                                              # logits: (B, C), target: (B,)

        log_probs = F.log_softmax(logits, dim=1)                                    # 각 클래스별 log-softmax 확률 계산
        probs = torch.exp(log_probs)                                                # log확률 → 확률 값으로 복원 (exp)

        ## --- one-hot target ---
        target_one_hot = F.one_hot(target, num_classes=logits.size(1)).float()
        pt = (probs * target_one_hot).sum(dim=1)                                    # 모델이 정답 클래스를 얼마나 확신했는지 pt 계산

        focal_weight = (1 - pt) ** self.gamma                                       # 정답 확률이 낮을수록 가중치가 커지는 focal weight 계산
        ce_loss = F.nll_loss(log_probs, target, reduction="none")                   # 기본 cross entropy 손실 계산(none: 각 샘플 loss 반환)
        loss = focal_weight * ce_loss                                               # focal weight를 적용하여 어려운 샘플에 더 큰 loss 부여

        ## --- batch 전체 평균 loss 반환 ---
        if self.reduction == "mean":
            return loss.mean()

        ## --- batch 전체 합 loss 반환 ---
        elif self.reduction == "sum":
            return loss.sum()

        ## --- reduction=None 이면 개별 loss 반환 ---
        return loss

# ===========================================================
# Helper Function: config 설정에 따라 사용할 loss function을 자동 선택
# ===========================================================
def get_loss():
    """
    config의 LOSS_NAME에 따라 loss 반환
    """

    ## --- 기본 cross entropy loss 반환 ---
    if LOSS_NAME == "ce":
        return nn.CrossEntropyLoss()

    ## --- label smoothing 적용된 CE loss 반환 ---
    elif LOSS_NAME == "ce_ls":
        # label smoothing CrossEntropy
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    ## --- focal loss 반환 ---
    elif LOSS_NAME == "focal":
        return FocalLoss(gamma=FOCAL_GAMMA)

    else:
        raise ValueError(f"지원하지 않는 LOSS_NAME: {LOSS_NAME}")