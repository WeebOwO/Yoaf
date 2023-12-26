def tood_build_config():
    return dict(
        backbone=dict(
            type="Unet", 
            n_blocks=[2, 3, 3, 3], 
            norm_type='BatchNorm', 
            act_type='ELU',
            coord=True),
        head=dict(
            type='TOOD',
            cls=1,
            in_channels=96,
            reg_max=8,
        ))

def basic_build_config():
    return dict(
        backbone=dict(
            type="Unet", 
            n_blocks=[2, 3, 3, 3], 
            norm_type='BatchNorm', 
            act_type='ELU',
            coord=True),
        head=dict(
            type='ClsRegHead',
            cls=1,
            in_channels=96,
            reg_max=8
        ))