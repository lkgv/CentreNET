import os
import configparser
import yaml

__all__ = ['Configures']

default_cfg_dir = 'config'
default_cfg_dir = os.path.abspath(os.path.expanduser(default_cfg_dir))


class Configures:
    def __init__(self, cfg='train.yml'):
        config_name = os.path.join(default_cfg_dir, cfg)
        config_file = open(config_name, 'r')
        self._config = yaml.load(config_file)

        # print configure infomation
        print('=' * 13 + 'Configure' + '=' * 13)
        with open(config_name, 'r') as config_file:
            configs = config_file.read()
            print(configs)
        print('=' * 35)

    def __call__(self, section, option):
        return self._config[section][option]


def checkdir(path):
    if os.path.isfile(path):
        print('current path {} is a file instead of dir'.format(path))
        raise TypeError('There have been a file in appointed path!')
    elif os.path.isdir(path):
        pass
    else:
        os.mkdir(path)