from config import get_args, Config
from train import train

def main(c: Config):
    print("CONFIG:",c)
    # model
    if c.action_type in ['training', 'testing']:
        c.model = f'{c.dataset}_{c.encoder_arch}_{c.decoder_arch}_pl{c.pool_layers}_cb{c.coupling_blocks}_inp{c.input_size}_run{c.run_name}_{c.class_name}'
        if c.action_type == 'training':
            train(c)
    else:
        raise NotImplementedError(f'Passed in running mode of {c.action_type} Please passin correct mode')


if __name__ == '__main__':
    configs = get_args()
    main(configs)