from config import get_args
from train import train

def main(c):
    # model
    if c.action_type in ['training', 'testing']:
        c.model = f'{c.dataset}_{c.enc_arch}_{c.dec_arch}_pl{c.pool_layers}_cb{c.coupling_blocks}_inp{c.input_size}_run{c.run_name}_{c.class_name}'
    else:
        raise NotImplementedError(f'Passed in running mode of {c.action_type} Please passin correct mode')


if __name__ == '__main__':
    configs = get_args()
    main(configs)