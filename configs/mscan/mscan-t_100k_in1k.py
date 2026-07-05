_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/kaggleimagenet_bs32.py',
]
# model settings
# checkpoint_file = '/kaggle/input/models/tmaitn/mscan-t-20230227-119e8c9f/other/default/1/mscan_t_20230227-119e8c9f.pth'  # noqa
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
# checkpoint_file = '~/.cache/torch/hub/checkpoints/mscan_t_20230227-119e8c9f.pth'  # noqa
model = dict(
    type='ImageClassifier',
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# dataset settings
train_dataloader = dict(batch_size=32)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.01, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # 'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    # dict(
        # type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        # begin=1500,
        begin=0,
        end=100000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 10k
max_iters = 100000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=max_iters),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
