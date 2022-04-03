import argparse
import sys
import operator

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kw):
        kw['formatter_class'] = SortingHelpFormatter
        argparse.ArgumentParser.__init__(self, *args, **kw)
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(1)

class SortingHelpFormatter(argparse.HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=operator.attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)