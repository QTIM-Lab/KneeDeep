import sys
from pkg_resources import get_distribution, DistributionNotFound
from argparse import ArgumentParser
from appdirs import AppDirs
import yaml
from os.path import join, isdir, isfile
from os import makedirs
from inference import localize_knees


try:
    __version__ = get_distribution('KneeDeep').version
except DistributionNotFound:
    __version__ = None


class KneeDeepCommands(object):

    def __init__(self):

        self.conf_dict, self.conf_file = initialize()

        parser = ArgumentParser(
            description='A set of commands for knee joint localization and classification from radiographs',
            usage='''kneedeep <command> [<args>]
            The following commands are available:
               configure                    Configure KneeDeep models to use for localization and classification
               localize_joints              Localize knee joints in supplied radiographs(s)
            ''')

        parser.add_argument('command', help='Command to run')
        args = parser.parse_args(sys.argv[1:2])

        if not hasattr(self, args.command):
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    def configure(self):

        parser = ArgumentParser(
            description='Update models for localization or anomaly detection')

        pass

    def localize_knees(self):

        parser = ArgumentParser(description='Perform knee joint localization')
        # TODO add arguments
        args = parser.parse_args()

        localize_knees(self.conf_dict, args.images, args.out_dir)


def initialize(localizer=None, classifier=None):

    # Setup appdirs
    dirs = AppDirs("KneeDeep", "QTIM", version=__version__)
    conf_dir = dirs.user_config_dir
    conf_file = join(conf_dir, 'config.yaml')

    if not isdir(conf_dir):
        makedirs(conf_dir)

    if not isfile(conf_file):
        config_dict = {'localizer': localizer, 'classifier': classifier}

        with open(conf_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    return yaml.load(open(conf_file, 'r')), conf_file


def main():
    KneeDeepCommands()
