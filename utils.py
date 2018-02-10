import os
import configparser

__all__ = ['Configures']

default_configFile = os.path.abspath(os.path.expanduser('config.ini'))

class Configures:
    def __init__(self, configfile=default_configFile):
        self._parser = configparser.ConfigParser()
        self._parser.read('%s'%configfile)

        print('===========Configure===========')
        for section in self._parser.sections():
            print('[{}]:'.format(section))
            for option in self._parser.options(section):
                print('\t{} = {}'.format(option, self._parser[section][option]))
        print('=================================')

    def __call__(self, section, option):
        return self._parser[section][option]
