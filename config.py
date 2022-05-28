from config_dlsim import Config_DLSIM
from config_dlsr import Config_DLSR
from config_dlsr2 import Config_DLSR2

class Config(Config_DLSIM):
    print('loading config')

if __name__ == '__main__':
    cfg = Config()
    print(cfg.dataset_name)