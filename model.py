from sub_models.models import wide_resnet50_2

def load_encoder_arch(config, num_pool_layers):
    # Load the encoder in eval mode (pretrained)
    pool_count = 0
    pool_dimension = list()
    pool_layers = [f'layer{i}' for i in range(num_pool_layers)]
    enc_arch = config.encoder_architecture
    if 'resnet' in enc_arch:
        if enc_arch in ['resnet18', 'resnet34', 'resnet50']:
            raise NotImplementedError()
        elif enc_arch != 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=True, progress= True)
        else:
            raise NotImplementedError()
        # TODO: Extract the intermediates layer feature maps
        if num_pool_layers >=3:
            encoder 