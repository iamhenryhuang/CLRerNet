import os
import warnings

import torch
import torch.nn as nn

from mmdet.registry import MODELS


@MODELS.register_module()
class ZeroDCEEnhancer(nn.Module):
    """Zero-DCE enhancer used as an online positive-view generator.

    This module follows the DCE-Net curve estimation structure and can load
    pretrained weights from a checkpoint.
    """

    def __init__(self, channels=32, pretrained=None, requires_grad=False):
        super(ZeroDCEEnhancer, self).__init__()
        self.requires_grad = bool(requires_grad)
        self._use_identity = False

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, channels, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(channels * 2, 24, 3, 1, 1)

        if pretrained:
            ckpt_path = pretrained
            if not os.path.isabs(ckpt_path):
                # Resolve relative path from project root as a fallback.
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', '..', '..')
                )
                root_relative = os.path.join(project_root, ckpt_path)
                if os.path.exists(root_relative):
                    ckpt_path = root_relative

            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                state_dict = ckpt.get('state_dict', ckpt)
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                expected_keys = {n for n, _ in self.named_parameters()}
                missing_critical = [k for k in missing if k in expected_keys]
                if missing_critical:
                    warnings.warn(
                        f"ZeroDCE checkpoint '{ckpt_path}' missing critical keys: "
                        f"{missing_critical}. Module may behave like an untrained net.",
                        RuntimeWarning,
                    )
                if unexpected:
                    warnings.warn(
                        f"ZeroDCE checkpoint '{ckpt_path}' has unexpected keys: "
                        f"{unexpected}",
                        RuntimeWarning,
                    )
            else:
                if not self.requires_grad:
                    self._use_identity = True
                warnings.warn(
                    f"ZeroDCE checkpoint not found: {pretrained}. "
                    "Fallback to identity enhancement.",
                    RuntimeWarning,
                )

        if not self.requires_grad:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, x):
        x = x.clamp(0.0, 1.0)
        if self._use_identity:
            return x

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # Official Zero-DCE curve: y = x + r*(x^2 - x) = x - r*(x*(1-x))
        # r is learned negative for low-light → brightens the image.
        y = x + r1 * (torch.pow(x, 2) - x)
        y = y + r2 * (torch.pow(y, 2) - y)
        y = y + r3 * (torch.pow(y, 2) - y)
        y = y + r4 * (torch.pow(y, 2) - y)
        y = y + r5 * (torch.pow(y, 2) - y)
        y = y + r6 * (torch.pow(y, 2) - y)
        y = y + r7 * (torch.pow(y, 2) - y)
        y = y + r8 * (torch.pow(y, 2) - y)
        return y.clamp(0.0, 1.0)