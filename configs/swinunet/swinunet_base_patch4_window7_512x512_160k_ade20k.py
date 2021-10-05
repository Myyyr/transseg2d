_base_ = [
    '../_base_/models/swin_unet.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        num_classes=150
    ),
    decode_head=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        num_classes=150
    )
    #,
    # auxiliary_head=dict(
    #     in_channels=512,
    #     num_classes=150
    # )
    )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)


gpu_ids = [2]


"""
raceback (most recent call last):
  File "tools/train.py", line 163, in <module>
    main()
  File "tools/train.py", line 159, in main
    meta=meta)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/apis/train.py", line 116, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 130, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/iter_based_runner.py", line 66, in train
    self.call_hook('after_train_iter')
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/base_runner.py", line 307, in call_hook
    getattr(hook, fn_name)(self)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/core/evaluation/eval_hooks.py", line 30, in after_train_iter
    results = single_gpu_test(runner.model, self.dataloader, show=False)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/apis/test.py", line 60, in single_gpu_test
    result = model(return_loss=False, **data)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/parallel/data_parallel.py", line 42, in forward
    return super().forward(*inputs, **kwargs)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/parallel/data_parallel.py", line 153, in forward
    return self.module(*inputs[0], **kwargs[0])
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmcv/runner/fp16_utils.py", line 84, in new_func
    return old_func(*args, **kwargs)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/base.py", line 124, in forward
    return self.forward_test(img, img_metas, **kwargs)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/base.py", line 106, in forward_test
    return self.simple_test(imgs[0], img_metas[0], **kwargs)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/encoder_decoder.py", line 265, in simple_test
    seg_logit = self.inference(img, img_meta, rescale)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/encoder_decoder.py", line 250, in inference
    seg_logit = self.whole_inference(img, img_meta, rescale)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/encoder_decoder.py", line 217, in whole_inference
    seg_logit = self.encode_decode(img, img_meta)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/encoder_decoder.py", line 87, in encode_decode
    x = self.extract_feat(img)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/segmentors/encoder_decoder.py", line 79, in extract_feat
    x = self.backbone(img)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/backbones/swin_unet_encoder.py", line 199, in forward
    x, x_downsample, Wh, Ww = self.forward_features(x)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/backbones/swin_unet_encoder.py", line 162, in forward_features
    x, Wh, Ww = layer(x, Wh, Ww)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/utils/swin_unet_utils.py", line 478, in forward
    x_down = self.downsample(x, H, W)
  File "/etudiants/siscol/t/themyr_l/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/nn/modules/module.py", line 722, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/etudiants/siscol/t/themyr_l/transseg2d/mmseg/models/utils/swin_unet_utils.py", line 322, in forward
    assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
AssertionError: x size (128*171) are not even
"""