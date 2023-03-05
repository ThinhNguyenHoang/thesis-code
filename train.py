import os, time
# Arguments
def encoding_module_eval(features, model):
    anomaly_feature = model.encoder.encode(features)
    # Calculate loglikelyhood and estimate density
    score = get_score(anomaly_feature)
    pass

def eval_one_batch(input, model):
    # Calculate the salient object:
    salient_obj_img  = model.saliency_detector.eval(input)
    features = model.feature_extrator.eval(input) # L feature map
    
    pass

def train(config):
    # Get the saliency image: (black and white)
    run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    pool_layers = config.pool_layers
    print('Number of pooling ', pool_layers)
    # Load encoder in evaluate mode
    encoder, pool_layers, pool_dimensions = load_encoder_arch(config, pool_layers)

    decoders =  [load_decoder_arch(config, pool_dimension) for pool_dimension in pool_dimensions]

    pass
