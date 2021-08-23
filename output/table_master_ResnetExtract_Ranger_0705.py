checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
alphabet_file = '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/train/textline_recognition_alphabet.txt'
alphabet_len = 3480
max_seq_len = 500
start_end_same = False
label_convertor = dict(
    type='TableMasterConvertor',
    dict_file=
    '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/train/textline_recognition_alphabet.txt',
    max_seq_len=500,
    start_end_same=False,
    with_unknown=True)
PAD = 3483
model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
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
        type='TableMasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(headers=8, d_model=512, dropout=0.0),
            src_attn=dict(headers=8, d_model=512, dropout=0.0),
            feed_forward=dict(d_model=512, d_ff=2024, dropout=0.0),
            size=512,
            dropout=0.0),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=3483, reduction='mean'),
    bbox_loss=dict(type='TableL1Loss', reduction='sum'),
    label_convertor=dict(
        type='TableMasterConvertor',
        dict_file=
        '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/train/textline_recognition_alphabet.txt',
        max_seq_len=500,
        start_end_same=False,
        with_unknown=True),
    max_seq_len=500)
TRAIN_STATE = True
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TableResize', keep_ratio=True, long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=True),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'scale_factor',
            'bbox', 'bbox_masks', 'pad_shape'
        ])
]
valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TableResize', keep_ratio=True, long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=True),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TableResize', keep_ratio=True, long_size=480),
    dict(
        type='TablePad',
        size=(480, 480),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'pad_shape'
        ])
]
dataset_type = 'OCRDataset'
train_img_prefix = '/home/zhaohj/Documents/dataset/Table/TAL/Table/images'
train_anno_file1 = '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/train/StructureLabelAddEmptyBbox_train'
train1 = dict(
    type='OCRDataset',
    img_prefix='/home/zhaohj/Documents/dataset/Table/TAL/Table/images',
    ann_file=
    '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/train/StructureLabelAddEmptyBbox_train',
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=500,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='TableResize', keep_ratio=True, long_size=480),
        dict(
            type='TablePad',
            size=(480, 480),
            pad_val=0,
            return_mask=True,
            mask_ratio=(8, 8),
            train_state=True),
        dict(type='TableBboxEncode'),
        dict(type='ToTensorOCR'),
        dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'filename', 'ori_shape', 'img_shape', 'text', 'scale_factor',
                'bbox', 'bbox_masks', 'pad_shape'
            ])
    ],
    test_mode=False)
valid_img_prefix = '/home/zhaohj/Documents/dataset/Table/TAL/Table/images'
valid_anno_file1 = '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/val/StructureLabelAddEmptyBbox_val'
valid = dict(
    type='OCRDataset',
    img_prefix='/home/zhaohj/Documents/dataset/Table/TAL/Table/images',
    ann_file=
    '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/val/StructureLabelAddEmptyBbox_val',
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=500,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='TableResize', keep_ratio=True, long_size=480),
        dict(
            type='TablePad',
            size=(480, 480),
            pad_val=0,
            return_mask=True,
            mask_ratio=(8, 8),
            train_state=True),
        dict(type='TableBboxEncode'),
        dict(type='ToTensorOCR'),
        dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'filename', 'ori_shape', 'img_shape', 'scale_factor',
                'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks',
                'pad_shape'
            ])
    ],
    dataset_info='table_master_dataset',
    test_mode=True)
test_img_prefix = '/home/zhaohj/Documents/dataset/Table/TAL/Table/images'
test_anno_file1 = '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/val/StructureLabelAddEmptyBbox_val'
test = dict(
    type='OCRDataset',
    img_prefix='/home/zhaohj/Documents/dataset/Table/TAL/Table/images',
    ann_file=
    '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/val/StructureLabelAddEmptyBbox_val',
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=500,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='TableResize', keep_ratio=True, long_size=480),
        dict(
            type='TablePad',
            size=(480, 480),
            pad_val=0,
            return_mask=True,
            mask_ratio=(8, 8),
            train_state=True),
        dict(type='ToTensorOCR'),
        dict(type='NormalizeOCR', mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=[
                'filename', 'ori_shape', 'img_shape', 'scale_factor',
                'img_norm_cfg', 'ori_filename', 'pad_shape'
            ])
    ],
    dataset_info='table_master_dataset',
    test_mode=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/home/zhaohj/Documents/dataset/Table/TAL/Table/images',
                ann_file=
                '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/train/StructureLabelAddEmptyBbox_train',
                loader=dict(
                    type='TableHardDiskLoader',
                    repeat=1,
                    max_seq_len=500,
                    parser=dict(
                        type='TableStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='TableResize', keep_ratio=True, long_size=480),
                    dict(
                        type='TablePad',
                        size=(480, 480),
                        pad_val=0,
                        return_mask=True,
                        mask_ratio=(8, 8),
                        train_state=True),
                    dict(type='TableBboxEncode'),
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
                            'scale_factor', 'bbox', 'bbox_masks', 'pad_shape'
                        ])
                ],
                test_mode=False)
        ]),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/home/zhaohj/Documents/dataset/Table/TAL/Table/images',
                ann_file=
                '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/val/StructureLabelAddEmptyBbox_val',
                loader=dict(
                    type='TableHardDiskLoader',
                    repeat=1,
                    max_seq_len=500,
                    parser=dict(
                        type='TableStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='TableResize', keep_ratio=True, long_size=480),
                    dict(
                        type='TablePad',
                        size=(480, 480),
                        pad_val=0,
                        return_mask=True,
                        mask_ratio=(8, 8),
                        train_state=True),
                    dict(type='TableBboxEncode'),
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
                            'scale_factor', 'img_norm_cfg', 'ori_filename',
                            'bbox', 'bbox_masks', 'pad_shape'
                        ])
                ],
                dataset_info='table_master_dataset',
                test_mode=True)
        ]),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='OCRDataset',
                img_prefix=
                '/home/zhaohj/Documents/dataset/Table/TAL/Table/images',
                ann_file=
                '/home/zhaohj/Documents/dataset/Table/TAL/Table/precessed_data/val/StructureLabelAddEmptyBbox_val',
                loader=dict(
                    type='TableHardDiskLoader',
                    repeat=1,
                    max_seq_len=500,
                    parser=dict(
                        type='TableStrParser',
                        keys=['filename', 'text'],
                        keys_idx=[0, 1],
                        separator=' ')),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='TableResize', keep_ratio=True, long_size=480),
                    dict(
                        type='TablePad',
                        size=(480, 480),
                        pad_val=0,
                        return_mask=True,
                        mask_ratio=(8, 8),
                        train_state=True),
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
                            'scale_factor', 'img_norm_cfg', 'ori_filename',
                            'pad_shape'
                        ])
                ],
                dataset_info='table_master_dataset',
                test_mode=True)
        ]))
optimizer = dict(type='Ranger', lr=0.001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.3333333333333333,
    step=[12, 15])
total_epochs = 200
evaluation = dict(interval=1, metric='acc')
fp16 = dict(loss_scale='dynamic')
work_dir = './output/'
gpu_ids = range(0, 1)
