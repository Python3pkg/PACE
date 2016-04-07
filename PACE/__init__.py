#!/usr/bin/env python3.5

import PACE
import os
import argparse
import matplotlib

matplotlib.style.use('ggplot')

DATASTORE = 'linefitdata.mat'
HEADER = (' ____   _    ____ _____\n'
          '|  _ \ / \  / ___| ____|\n'
          '| |_) / _ \| |   |  _|\n'
          '|  __/ ___ \ |___| |___\n'
          '|_| /_/   \_\____|_____|\n\n'
          'PACE: Parameterization & Analysis of Conduit Edges\n'
          'William Farmer - 2015\n')


def main():
    args = get_args()
    data = PACE.DataStore(DATASTORE)
    data.load()

    # Establish directory for img outputs
    if not os.path.exists('./img'):
        os.makedirs('./img')

    if args.plot:
        for filename in args.files:
            print('Plotting ' + filename)
            plot_name = './img/' + filename + '.general_fit.png'
            fit = PACE.LineFit(filename)
            fit.plot_file(name=plot_name, time=args.time)
    if args.analyze:
        for filename in args.files:
            PACE.manage_file_analysis(args, filename, data)
    if args.plotdata:
        data.plot_traindata()
    if args.machinetest:
        learner = ML(algo=args.model)
    if args.printdata:
        data.printdata()
    if args.printdatashort:
        data.printshort()


def get_args() -> argparse.Namespace:
    """
    Get program arguments.

    Just use --help....
    """
    parser = argparse.ArgumentParser(prog='python3 linefit.py',
                                     description=('Parameterize and analyze '
                                                  'usability of conduit edge data'))
    parser.add_argument('files', metavar='F', type=str, nargs='*',
                        help=('File(s) for processing. '
                              'Each file has a specific format: '
                              'See README (or header) for specification.'))
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help=('Create Plot of file(s)? Note, unless --time flag used, '
                              'will plot middle time.'))
    parser.add_argument('-pd', '--plotdata', action='store_true', default=False,
                        help='Create plot of current datastore.')
    parser.add_argument('-a', '--analyze', action='store_true', default=False,
                        help=('Analyze the file and determine Curvature/Noise parameters. '
                              'If --time not specified, will examine entire file. '
                              'This will add results to datastore with false flags '
                              'in accept field if not provided.'))
    parser.add_argument('-mt', '--machinetest', action='store_true', default=False,
                        help=('Determine if the times from the file are usable based on '
                              'supervised learning model. If --time not specified, '
                              'will examine entire file.'))
    parser.add_argument('-m', '--model', type=str, default='nn',
                        help=('Learning Model to use. Options are ["nn", "svm", "forest", "sgd"]'))
    parser.add_argument('-nnk', '--nnk', type=int, default=10,
                        help=('k-Parameter for k nearest neighbors. Google it.'))
    parser.add_argument('-t', '--time', type=int, default=None,
                        help=('Time (column) of data to use for analysis OR plotting. '
                              'Zero-Indexed'))
    parser.add_argument('-d', '--datastore', type=str, default=DATASTORE,
                        help=("Datastore filename override. "
                              "Don't do this unless you know what you're doing"))
    parser.add_argument('-pds', '--printdata', action='store_true', default=False,
                        help=("Print data"))
    parser.add_argument('-pdss', '--printdatashort', action='store_true', default=False,
                        help=("Print data short"))
    args = parser.parse_args()
    return args
