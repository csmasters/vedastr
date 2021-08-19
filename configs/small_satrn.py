###############################################################################
# 1. deploy
import dict as dict

size = (32, 100)
mean, std = 0.5, 0.5

sensitive = True
character = ' !"#$%&"'"()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥§©­®°³"'·»¼½¾ÀÁÅÇÈÉÍÑÓ×ÚÜàáâãäçèéêìíîïñòóôöøúüýĄęİıłńœŞşǵəɪʃʊ˚ΛβμЙй฿–—‘’“”„†•…※⁶⁺⁻₁₂€₹℃™⅓⅔⅛ⅠⅡⅢⅣⅤⅥⅦⅧⅩ←↑→↓√∧∨≥①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮ⓥ┃■□▪▲▶►▼◀◆◇○◎●★☆☑☺♡✓✕✱✳✶⠁、。《》「」〔〕一丨口️！＄％＆＇（），－．／：；＝？］｜～￡￥'  # need character
test_sensitive = True
test_character = ' !"#$%&"'"()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz{|}~¡¢£¤¥§©«­®°³"'·»¼½¾ÀÁÅÇÈÉÍÑÓ×ÚÜàáâãäçèéêìíîïñòóôöøúüýĄęİıłńœŞşǵəɪʃʊ˚ΛβμЙй฿–—‘’“”„†•…※⁶⁺⁻₁₂€₹℃™⅓⅔⅛ⅠⅡⅢⅣⅤⅥⅦⅧⅩ←↑→↓√∧∨≥①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮ⓥ┃■□▪▲▶►▼◀◆◇○◎●★☆☑☺♡✓✕✱✳✶⠁、。《》「」〔〕一丨口️！＄％＆＇（），－．／：；＝？］｜～￡￥'
batch_max_length = 99

dropout = 0.1
n_e = 9
n_d = 3
hidden_dim = 64
n_head = 8
batch_norm = dict(type='BN')
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)
num_class = len(character) + 1
num_steps = batch_max_length + 1

deploy = dict(
    transform=[
        dict(type='Sensitive', sensitive=sensitive, need_character=character),
        dict(type='ToGray'),
        dict(type='Resize', size=size),
        dict(type='Normalize', mean=mean, std=std),
        dict(type='ToTensor'),
    ],
    converter=dict(
        type='AttnConverter',
        character=character,
        batch_max_length=batch_max_length,
        go_last=True,
    ),
    model=dict(
        type='GModel',
        need_text=True,
        body=dict(
            type='GBody',
            pipelines=[
                dict(
                    type='FeatureExtractorComponent',
                    from_layer='input',
                    to_layer='cnn_feat',
                    arch=dict(
                        encoder=dict(
                            backbone=dict(
                                type='GResNet',
                                layers=[
                                    ('conv',
                                     dict(type='ConvModule', in_channels=1, out_channels=int(hidden_dim / 2),
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('conv',
                                     dict(type='ConvModule', in_channels=int(hidden_dim / 2), out_channels=hidden_dim,
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                    ('block',
                                     dict(block_name='BasicBlock', planes=int(hidden_dim * 2), blocks=1, stride=1)),
                                    ('conv',
                                     dict(type='ConvModule', in_channels=int(hidden_dim * 2),
                                          out_channels=int(hidden_dim),
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('conv',
                                     dict(type='ConvModule', in_channels=int(hidden_dim),
                                          out_channels=int(hidden_dim / 2),
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('conv',
                                     dict(type='ConvModule', in_channels=int(hidden_dim / 2), out_channels=int(hidden_dim * 4),
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                    ('block', dict(block_name='BasicBlock', planes=int(hidden_dim * 4), blocks=2, stride=1)),
                                    ('conv',
                                     dict(type='ConvModule', in_channels=int(hidden_dim * 4), out_channels=int(hidden_dim / 2),
                                          kernel_size=3,
                                          stride=1, padding=1, norm_cfg=batch_norm)),
                                ],
                            ),
                        ),
                        collect=dict(type='CollectBlock', from_layer='c2'),
                    ),
                ),
                dict(
                    type='SequenceEncoderComponent',
                    from_layer='cnn_feat',
                    to_layer='src',
                    arch=dict(
                        type='TransformerEncoder',
                        position_encoder=dict(
                            type='Adaptive2DPositionEncoder',
                            in_channels=hidden_dim,
                            max_h=100,
                            max_w=100,
                            dropout=dropout,
                        ),
                        encoder_layer=dict(
                            type='TransformerEncoderLayer2D',
                            attention=dict(
                                type='MultiHeadAttention',
                                in_channels=hidden_dim,
                                k_channels=hidden_dim // n_head,
                                v_channels=hidden_dim // n_head,
                                n_head=n_head,
                                dropout=dropout,
                            ),
                            attention_norm=layer_norm,
                            feedforward=dict(
                                type='Feedforward',
                                layers=[
                                    dict(type='ConvModule', in_channels=hidden_dim, out_channels=hidden_dim * 4,
                                         kernel_size=3, padding=1,
                                         bias=True, norm_cfg=None, activation='relu', dropout=dropout),
                                    dict(type='ConvModule', in_channels=hidden_dim * 4, out_channels=hidden_dim,
                                         kernel_size=3, padding=1,
                                         bias=True, norm_cfg=None, activation=None, dropout=dropout),
                                ],
                            ),
                            feedforward_norm=layer_norm,
                        ),
                        num_layers=n_e,
                    ),
                ),
            ],
        ),
        head=dict(
            type='TransformerHead',
            src_from='src',
            num_steps=num_steps,
            pad_id=num_class,
            decoder=dict(
                type='TransformerDecoder',
                position_encoder=dict(
                    type='PositionEncoder1D',
                    in_channels=hidden_dim,
                    max_len=100,
                    dropout=dropout,
                ),
                decoder_layer=dict(
                    type='TransformerDecoderLayer1D',
                    self_attention=dict(
                        type='MultiHeadAttention',
                        in_channels=hidden_dim,
                        k_channels=hidden_dim // n_head,
                        v_channels=hidden_dim // n_head,
                        n_head=n_head,
                        dropout=dropout,
                    ),
                    self_attention_norm=layer_norm,
                    attention=dict(
                        type='MultiHeadAttention',
                        in_channels=hidden_dim,
                        k_channels=hidden_dim // n_head,
                        v_channels=hidden_dim // n_head,
                        n_head=n_head,
                        dropout=dropout,
                    ),
                    attention_norm=layer_norm,
                    feedforward=dict(
                        type='Feedforward',
                        layers=[
                            dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 4, bias=True,
                                 activation='relu', dropout=dropout),
                            dict(type='FCModule', in_channels=hidden_dim * 4, out_channels=hidden_dim, bias=True,
                                 activation=None, dropout=dropout),
                        ],
                    ),
                    feedforward_norm=layer_norm,
                ),
                num_layers=n_d,
            ),
            generator=dict(
                type='Linear',
                in_features=hidden_dim,
                out_features=num_class,
            ),
            embedding=dict(
                type='Embedding',
                num_embeddings=num_class + 1,
                embedding_dim=hidden_dim,
                padding_idx=num_class,
            ),
        ),
    ),
    postprocess=dict(
        sensitive=test_sensitive,
        character=test_character,
    ),
)

###############################################################################
# 2.common

common = dict(
    seed=1111,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=True,
    cudnn_benchmark=True,
    metric=dict(type='Accuracy'),
)

###############################################################################
dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter=True,
    character=character,
)
test_dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter=False,
    character=test_character,
)

data_root = '../../../data/'

###############################################################################
# 3. test

batch_size = 320
# data
# test_root = data_root + 'evaluation/'
test_root = '/home/xinyudong/data/experiment/eval/'
# test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077',
#                      'IIIT5k_3000', 'SVT', 'SVTP']
# test_dataset = [dict(type='LmdbDataset', root=test_root + f_name,
#                      **test_dataset_params) for f_name in test_folder_names]
test_dataset = [(dict(type='LmdbDataset', root=test_root))]

test = dict(
    data=dict(
        dataloader=dict(
            type='DataLoader',
            batch_size=batch_size,
            num_workers=100,
            shuffle=True,
        ),
        dataset=test_dataset,
        transform=[
            dict(type='Sensitive', sensitive=test_sensitive, need_character=test_character),
            dict(type='ToGray'),
            dict(type='Resize', size=size),
            dict(type='Normalize', mean=mean, std=std),
            dict(type='ToTensor'),
        ],
    ),
    postprocess_cfg=dict(
        sensitive=test_sensitive,
        character=test_character,
    ),
)

###############################################################################
# 4. train

root_workdir = '../../../data/workdir'  # save directory

# data
train_root = data_root + 'experiment/'
# MJ dataset
train_root_f = train_root + 'train0/'
# mj_folder_names = ['/MJ_test', 'MJ_valid', 'MJ_train']
# ST dataset
train_root_s = train_root + 'train1/'

# train_dataset_mj = [dict(type='LmdbDataset', root=train_root_f + folder_name)
#                     for folder_name in mj_folder_names]
train_dataset_f = [dict(type='LmdbDataset', root=train_root_f)]
train_dataset_s = [dict(type='LmdbDataset', root=train_root_s)]

# valid
valid_root = train_root + 'val/'
valid_dataset = dict(type='LmdbDataset', root=valid_root, **test_dataset_params)

train_transforms = [
    dict(type='Sensitive', sensitive=sensitive, need_character=character),
    dict(type='ToGray'),
    # dict(type='ExpandRotate', limit=34, p=0.5),
    dict(type='Resize', size=size),
    dict(type='Normalize', mean=mean, std=std),
    dict(type='ToTensor'),
]

max_epochs = 100
# milestones = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]  # epoch start from 0, so 2 means lr decay at 3 epoch, 4 means lr decay at the end of

train = dict(
    data=dict(
        train=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=batch_size,
                num_workers=32,
            ),
            sampler=dict(
                type='BalanceSampler',
                batch_size=batch_size,
                shuffle=True,
                oversample=True,
            ),
            dataset=dict(
                type='ConcatDatasets',
                datasets=[
                    dict(
                        type='ConcatDatasets',
                        datasets=train_dataset_f,
                    ),
                    dict(
                        type='ConcatDatasets',
                        datasets=train_dataset_s,
                    )
                ],
                batch_ratio=[0.3, 0.7],
                **dataset_params,
            ),
            transform=train_transforms,
        ),
        val=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=batch_size,
                num_workers=32,
                shuffle=False,
            ),
            dataset=valid_dataset,
            transform=test['data']['transform'],
        ),
    ),
    optimizer=dict(type='Adam', lr=0.1),
    criterion=dict(type='CrossEntropyLoss', ignore_index=num_class),
    lr_scheduler=dict(type='ExponentialLR',
                      iter_based=True,
                      warmup_epochs=1,
                      gamma=0.9,
                      niter_per_epoch=1000,
                      step=10
                      ),
    max_epochs=max_epochs,
    log_interval=10,
    trainval_ratio=2000,
    snapshot_interval=10000,
    save_best=True,
    resume=False,
)
