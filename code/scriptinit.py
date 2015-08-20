__author__ = 'Kelvin Gu'

from os.path import dirname, abspath, join, basename
import sys

current_dir = dirname(abspath(__file__))

if basename(current_dir) == 'code':
    # relevant files are already on path
    pass
elif basename(current_dir) == 'scripts':
    # relevant files are in sibling directory
    code_dir = join(dirname(current_dir), 'code')
    sys.path.insert(0, code_dir)
else:
    raise RuntimeError('Could not find code directory from {}.'.format(current_dir))

import warnings
import matplotlib
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    matplotlib.use('Agg')  # needed when running from server
