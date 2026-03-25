import torch
import torch.nn.functional as F
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.registry import MODELS


@MODELS.register_module()
class CLRerNet(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        sgm_loss_weight=0.5,  # 接收 config 的 0.5
        sgm_dark_threshold=0.45,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        """CLRerNet detector."""
        super(CLRerNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, data_preprocessor, init_cfg
        )
        self.sgm_loss_weight = float(sgm_loss_weight)  # 儲存權重
        self.sgm_dark_threshold = float(sgm_dark_threshold)

    def _build_scene_targets(self, img, img_metas, device, dtype):
        """Build binary scene labels.

        Priority:
        1) Use ``scene_label`` from ``img_metas`` when available.
        2) Fallback to brightness-based pseudo labels from input image.
        """
        gt_scene = []
        has_meta_labels = True
        for meta in img_metas:
            if isinstance(meta, dict) and "scene_label" in meta:
                v = meta["scene_label"]
                if isinstance(v, torch.Tensor):
                    v = v.detach().float().mean().item()
                elif isinstance(v, (list, tuple)):
                    v = float(v[0]) if len(v) > 0 else 0.0
                else:
                    v = float(v)
                gt_scene.append(v)
            else:
                has_meta_labels = False
                break

        if has_meta_labels and len(gt_scene) == img.size(0):
            return torch.as_tensor(gt_scene, device=device, dtype=dtype)

        # brightness pseudo label: dark=1, bright=0
        x = img.detach().float()
        if x.max() > 1.5:
            x = x / 255.0
        mean_luma = x.mean(dim=[1, 2, 3])
        pseudo_night = (mean_luma < self.sgm_dark_threshold).to(dtype=dtype)
        return pseudo_night.to(device=device)

    def _compute_sgm_loss(self, x, img, img_metas):
        """Compute Scene Aware Gate BCE loss."""
        if self.sgm_loss_weight <= 0:
            return None

        p = getattr(self.neck, 'sgm_p', None)
        if p is None:
            # fallback: derive a stable probability from first FPN output
            p_global = torch.sigmoid(x[0].mean(dim=[1, 2, 3]))
        else:
            p_global = p.reshape(p.size(0), -1).mean(dim=1)

        gt_scene = self._build_scene_targets(
            img, img_metas, device=p_global.device, dtype=p_global.dtype
        )

        if gt_scene.numel() != p_global.numel():
            return None

        loss_sgm = F.binary_cross_entropy(p_global, gt_scene)
        return self.sgm_loss_weight * loss_sgm

    def loss(self, batch_inputs, batch_data_samples):
        """MMDet3 training API."""
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)

        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        loss_sgm = self._compute_sgm_loss(x, batch_inputs, img_metas)
        if loss_sgm is not None:
            losses['loss_sgm'] = loss_sgm

        return losses

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas)

        # keep compatibility for legacy train path
        loss_sgm = self._compute_sgm_loss(x, img, img_metas)
        if loss_sgm is not None:
            losses['loss_sgm'] = loss_sgm

        return losses

    def predict(self, img, data_samples, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            data_samples (List[:obj:`DetDataSample`]): The data samples
                that include meta information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        for i in range(len(data_samples)):
            data_samples[i].metainfo["batch_input_shape"] = tuple(img.size()[-2:])

        x = self.extract_feat(img)
        outputs = self.bbox_head.predict(x, data_samples)
        return outputs
