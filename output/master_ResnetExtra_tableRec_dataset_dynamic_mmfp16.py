checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
alphabet_file = '/home/zhaohj/Documents/dataset/Table/TAL/processed_data/textline_recognition_alphabet.txt'
alphabet_len = 3694
max_seq_len = 100
start_end_same = False
label_convertor = dict(
    type='MasterConvertor',
    dict_file=
    '/home/zhaohj/Documents/dataset/Table/TAL/processed_data/textline_recognition_alphabet.txt',
    max_seq_len=100,
    start_end_same=False,
    with_unknown=True)
PAD = 3697
model = dict(
    type='MASTER',
    backbone=dict(
        type='ResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type='channel_add',
            layers=[False, True, True, True]),
        layers=[1, 2, 5, 3]),
    encoder=dict(
        type='PositionalEncoding', d_model=512, dropout=0.2, max_len=5000),
    decoder=dict(
        type='MasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(headers=8, d_model=512, dropout=0.0),
            src_attn=dict(headers=8, d_model=512, dropout=0.0),
            feed_forward=dict(d_model=512, d_ff=2024, dropout=0.0),
            size=512,
            dropout=0.0),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=3697, reduction='mean'),
    label_convertor=dict(
        type='MasterConvertor',
        dict_file=
        '/home/zhaohj/Documents/dataset/Table/TAL/processed_data/textline_recognition_alphabet.txt',
        max_seq_len=100,
        start_end_same=False,
        with_unknown=True),
    max_seq_len=100)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=256,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=64,
        min_width=64,
        max_width=256,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'valid_ratio',
            'img_norm_cfg', 'ori_filename'
        ])
]
dataset_type = 'OCRDataset'
img_prefix1 = ''
train_anno_file1 = '/data_0/pubtabnet/processed_data/recognition_val.txt'
train1 = dict(
    type='OCRDataset',
    img_prefix='',
    ann_file='/data_0/pubtabnet/processed_data/recognition_val.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='TableTextLineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator='	')),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='ResizeOCR',
            height=64,
            min_width=64,
            max_width=256,
            keep_aspect_ratio=True),
        dict(type='ToTensorOCR'),
        dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
            ])
    ],
    test_mode=False)
test_img_prefix = ''
test_ann_files = dict(
    table_Rec_val_debug_0='/data_0/pubtabnet/processed_data/recognition_val.txt'
)
testset = [
    dict(
        type='OCRDataset',
        img_prefix='',
        ann_file='/data_0/pubtabnet/processed_data/recognition_val.txt',
        loader=dict(
            type='HardDiskLoader',
            repeat=1,
            parser=dict(
                type='TableTextLineStrParser',
                keys=['filename', 'text'],
                keys_idx=[0, 1],
                separator='	')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeOCR',
                height=64,
                min_width=64,
                max_width=256,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(
                type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'img_norm_cfg', 'ori_filename'
                ])
        ],
        dataset_info='table_Rec_val_debug_0',
        test_mode=True)
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='',
                ann_file='/data_0/pubtabnet/processed_data/recognition_val.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='TableTextLineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='ResizeOCR',
                        height=64,
                        min_width=64,
                        max_width=256,
                        keep_aspect_ratio=True),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape', 'text',
                            'valid_ratio'
                        ])
                ],
                test_mode=False)
        ]),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='',
                ann_file='/data_0/pubtabnet/processed_data/recognition_val.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='TableTextLineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='ResizeOCR',
                        height=64,
                        min_width=64,
                        max_width=256,
                        keep_aspect_ratio=True),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'img_norm_cfg', 'ori_filename'
                        ])
                ],
                dataset_info='table_Rec_val_debug_0',
                test_mode=True)
        ]),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix='',
                ann_file='/data_0/pubtabnet/processed_data/recognition_val.txt',
                loader=dict(
                    type='HardDiskLoader',
                    repeat=1,
                    parser=dict(
                        type='TableTextLineStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator='	')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='ResizeOCR',
                        height=64,
                        min_width=64,
                        max_width=256,
                        keep_aspect_ratio=True),
                    dict(type='ToTensorOCR'),
                    dict(
                        type='NormalizeOCR',
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
                    dict(
                        type='Collect',
                        keys=['img'],
                        meta_keys=[
                            'filename', 'ori_shape', 'img_shape',
                            'valid_ratio', 'img_norm_cfg', 'ori_filename'
                        ])
                ],
                dataset_info='table_Rec_val_debug_0',
                test_mode=True)
        ]))
optimizer = dict(type='Ranger', lr=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    step=[5, 8])
total_epochs = 10
evaluation = dict(interval=1, metric='acc')
fp16 = dict(loss_scale='dynamic')
work_dir = './output/'
gpu_ids = range(0, 1)
