import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FocalLoss(nn.Module):
    """ Penalizes easy negatives to reduce high false-positive spikes """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()

class DualBackboneOncoGuard(nn.Module):
    """
    Dual-Stage Ensemble Architecture
    Ensembles DenseNet-201 (dense spatial connections) + EfficientNet-B4 (multi-scale features)
    """
    def __init__(self, pretrained=True):
        super(DualBackboneOncoGuard, self).__init__()
        
        # Backbone 1: DenseNet201
        self.densenet = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT if pretrained else None)
        in_dn = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        
        # Backbone 2: EfficientNet-B4
        self.efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None)
        in_en = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()
        
        # Multi-Head Stacking Classifier
        self.fusion_head = nn.Sequential(
            nn.Linear(in_dn + in_en, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs raw uncalibrated logit
        )

    def forward(self, x):
        feat_dn = self.densenet(x)
        feat_en = self.efficientnet(x)
        combined = torch.cat((feat_dn, feat_en), dim=1)
        return self.fusion_head(combined)

class CalibratedInferenceEngine:
    """ Applies Platt Scaling Calibration to raw neural outputs """
    def __init__(self, model, platt_a=1.0, platt_b=0.0):
        self.model = model
        self.platt_a = platt_a
        self.platt_b = platt_b

    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            raw_logit = self.model(input_tensor).item()
            # Calibrated Logit transformation
            calibrated_logit = (self.platt_a * raw_logit) + self.platt_b
            calibrated_prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
            
            # Clinical Priority Tiering Rules
            if calibrated_prob >= 0.85:
                priority = "🔴 PRIORITY 1: DEFINITIVE MALIGNANCY (URGENT)"
                sla = "< 24 Hours"
            elif calibrated_prob >= 0.35:
                priority = "🟡 PRIORITY 2: SUSPICIOUS / INDETERMINATE"
                sla = "< 48 Hours"
            else:
                priority = "🟢 PRIORITY 3: DEFINITIVE BENIGN / ROUTINE"
                sla = "Standard Queue"
                
            return {
                "calibrated_probability": round(float(calibrated_prob), 4),
                "priority_code": priority,
                "target_sla": sla
            }
