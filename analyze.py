#!/usr/bin/env python3

"""
 ____   _    ____ _____
|  _ \ / \  / ___| ____|
| |_) / _ \| |   |  _|
|  __/ ___ \ |___| |___
|_| /_/   \_\____|_____|

PACE: Parameterization & Analysis of Conduit Edges
William Farmer - 2015

TODO:
    * model training/testing
        * more models (technically)
    * multithreading

"""

import sys, os
import argparse
import hashlib

from tqdm import tqdm

import numpy as np
import numpy.linalg as linalg

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import scipy.integrate as si
import scipy.io as sco

import sklearn as sk
from sklearn import svm
from sklearn import preprocessing
from sklearn import neighbors


matplotlib.style.use('ggplot')


DATASTORE = 'linefitdata.mat'


def main():
    print((' ____   _    ____ _____\n'
           '|  _ \ / \  / ___| ____|\n'
           '| |_) / _ \| |   |  _|\n'
           '|  __/ ___ \ |___| |___\n'
           '|_| /_/   \_\____|_____|\n\n'
           'PACE: Parameterization & Analysis of Conduit Edges\n'
           'William Farmer - 2015\n'))
    args = get_args()
    data = DataStore(DATASTORE)
    data.load()

    # Establish directory for img outputs
    if not os.path.exists('./img'):
        os.makedirs('./img')

    if args.plot:
        for filename in args.files:
            print('Plotting ' + filename)
            plot_name = './img/' + filename + '.general_fit.png'
            fit = LineFit(filename)
            fit.plot_file(name=plot_name, t=args.time)
    if args.analyze:
        for filename in args.files:
            manage_file_analysis(args, filename, data)
    if args.plotdata:
        data.plot_traindata()
    if args.machinetest:
        ml = ML(algo=args.model)
    if args.printdata:
        data.printdata()
    if args.printdatashort:
        data.printshort()


def manage_file_analysis(args, filename, data):
    key = DataStore.hashfile(filename)
    print('Analyzing {} --> {}'.format(filename, key))
    if data.check_key(key):   # if exists in database, prepopulate
        fit = LineFit(filename, data=data.get_data(key))
    else:
        fit = LineFit(filename)
    if args.time:
        noise, curvature, rnge, domn = fit.analyze(t=args.time)
        newrow = [args.time, noise, curvature,
                rnge, domn, fit.accepts[args.time]]
        data.update1(key, newrow, len(fit.noises))
    else:
        fit.analyze_full()
        newrows = np.array([range(len(fit.noises)), fit.noises,
                         fit.curves, fit.ranges, fit.domains, fit.accepts])
        data.update(key, newrows)
    data.save()


class DataStore(object):
    def __init__(self, name):
        """
        Uses a .mat as datastore for compatibility.

        Eventually may want to switch to SQLite, or some database?  Not sure if
        ever needed. This class provides that extensible API structure however.

        Datafile has the following structure:

        learning_data = {filehash:[[trial_index, noise, curvature,
                                    range, domain, accept, viscosity]
                            ,...],...}

        Conveniently, you can use the domain field as a check as to whether or
        not the row has been touched. If domain=0 (for that row) then that
        means that it hasn't been updated.

        :param: str->name

        :return: DataStore Object
        """
        self.name = name

    def load(self):
        """
        Load datafile

        :return: None
        """
        try:
            self.data = sco.loadmat(self.name)
        except FileNotFoundError:
            self.data = {}

    def save(self):
        """
        Save datafile to disk

        :return: None
        """
        sco.savemat(self.name, self.data)

    def get_data(self, key):
        """
        Returns the specified data. Warning, ZERO ERROR HANDLING

        :param: str->key

        :return: array->2d data array
        """
        return self.data[key]

    def get_keys(self):
        """
        Return list of SHA512 hash keys that exist in datafile

        :return: list->list of strings
        """
        keys = []
        for key in self.data.keys():
            if key not in ['__header__', '__version__', '__globals__']:
                keys.append(key)
        return keys

    def check_key(self, key):
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: str->SHA512 hash

        :return: bool->whether or not exists
        """
        keys = self.get_keys()
        return (key in keys)

    def get_traindata(self):
        """
        Pulls all available data and concatenates for model training

        :return: array->2d array of points
        """
        traindata = None
        for key, value in self.data.items():
            if key not in ['__header__', '__version__', '__globals__']:
                if traindata is None:
                    traindata = value[np.where(value[:, 4] != 0)]
                else:
                    traindata = np.concatenate((traindata, value[np.where(value[:, 4] != 0)]))
        return traindata

    def plot_traindata(self, name='dataplot'):
        """
        Plots traindata.... choo choo...
        """
        traindata = self.get_traindata()

        plt.figure(figsize=(16, 16))
        plt.scatter(traindata[:, 1], traindata[:, 2], c=traindata[:, 5], marker='o', label='Datastore Points')
        plt.xlabel(r'$\log_{10}$ Noise')
        plt.ylabel(r'$\log_{10}$ Curvature')
        plt.legend(loc=2, fontsize='xx-large')
        plt.savefig('./img/{}.png'.format(name))

    def printdata(self):
        np.set_printoptions(threshold=np.nan)
        print(self.data)
        np.set_printoptions(threshold=1000)

    def printshort(self):
        print(self.data)

    def update(self, key, data):
        self.data[key] = data

    def update1(self, key, data, size):
        print(data)
        if key in self.get_keys():
            self.data[key][data[0]] = data
        else:
            newdata = np.zeros((size, 6))
            newdata[data[0]] = data
            self.data[key] = newdata

    @staticmethod
    def hashfile(name):
        # http://stackoverflow.com/questions/3431825/generating-a-md5-checksum-of-a-file
        # Using SHA512 for long-term support (hehehehe)
        hasher = hashlib.sha512()
        with open(name, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


class LineFit(object):
    def __init__(self, filename, data=None, function_number=16, spread_number=22):
        """
        Main class for line fitting and parameter determination

        :param: str->filename
        :param: (optional)int->number of functions
        :param: (optional)float->gaussian spread number

        :return: LineFit Object
        """
        self.filename                 = filename
        (self.averagedata, self.times,
            self.accepts, self.ratio, self.viscosity) = self._loadedges()
        self.domain                   = np.arange(len(self.averagedata[:, 0]))
        self.function_number          = function_number
        self.spread_number            = spread_number
        if data is None:
            self.noises  = np.zeros(len(self.times))
            self.curves  = np.zeros(len(self.times))
            self.ranges  = np.zeros(len(self.times))
            self.domains = np.zeros(len(self.times))
        else:
            self.noises  = data[:, 1]
            self.curves  = data[:, 2]
            self.ranges  = data[:, 3]
            self.domains = data[:, 4]

    def _loadedges(self):
        """
        Attempts to intelligently load the .mat file and take average of left and right edges

        :return: array->left and right averages
        :return: array->times for each column
        :return: array->accept/reject for each column
        :return: float->pixel-inch ratio
        """
        data = sco.loadmat(self.filename)
        datakeys = [k for k in data.keys()
                    if ('right' in k) or ('left' in k) or ('edge' in k)]
        averagedata = ((data[datakeys[0]] + data[datakeys[1]]) / 2)

        try:
            times = (data['times'] - data['times'].min())[0]
        except KeyError:
            times = np.arange(len(data[datakeys[0]][0]))

        try:
            accept = data['accept']
        except KeyError:
            accept = np.zeros(len(times))

        try:
            ratio  = data['ratio']
        except KeyError:
            ratio = 1

        try:
            viscosity = data['viscosity']
        except KeyError:
            viscosity = np.ones(len(times))
        return averagedata, times, accept, ratio, viscosity

    def plot_file(self, name=None, t=None):
        """
        Plot specific time for provided datafile.
        If no time provided, will plot middle.

        :param: (optional)str->savefile name
        :param: (optional)int->time/data column

        :return: None
        """
        if not t:
            t = int(len(self.times) / 2)
        if not name:
            name = './img/' + self.filename + '.png'
        yhat, r, r_hat, s = self._get_fit(t)
        plt.figure()
        plt.scatter(self.domain, self.averagedata[:, t], alpha=0.2)
        plt.plot(yhat)
        plt.savefig(name)

    @staticmethod
    def ddiff(arr):
        """
        Helper Function: Divided Differences

        input: array
        """
        return arr[:-1] - arr[1:]

    def _gaussian_function(self, n, x, b, i):
        """
        i'th Regression Model Gaussian

        :param: int->len(x)
        :param: array->x values
        :param: float->height of gaussian
        :param: int->position of gaussian

        :return: array->gaussian's over domain
        """
        return b * np.exp(-(1 / (self.spread_number * n)) * (x - ((n / self.function_number) * i))**2)

    def _get_fit(self, t):
        """
        Fit regression model to data

        :param: int->time (column of data)

        :return: array->predicted points
        :return: array->residuals
        :return: float->mean residual
        :return: float->error
        """
        y = self.averagedata[:, t]
        x = np.arange(len(y))
        n = len(x)
        X = np.zeros((n, self.function_number + 2))
        X[:, 0] = 1
        X[:, 1] = x
        for i in range(self.function_number):
            X[:, 2 + i] = self._gaussian_function(n, x, 1, i)
        betas = linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(y))
        y_hat = X.dot(betas)
        r = y - y_hat
        s = np.sqrt(r.transpose().dot(r) / (n - (self.function_number + 2)))
        return y_hat, r, r.mean(), s

    def _get_noise(self, r):
        """
        Determine Noise of Residuals.

        :param: array->residuals

        :return: float->noise
        """
        return np.mean(np.abs(r))

    def analyze(self, t=None):
        """
        Determine noise, curvature, range, and domain of specified array.

        :param: float->pixel to inch ratio
        :param: (optional)int->time (column) to use.

        :return: float->curvature
        :return: float->noise
        :return: float->range
        :return: float->domain
        """
        if not t:
            t = int(len(self.times) / 2)
        if self.domains[t] == 0:
            yhat, r, r_hat, s = self._get_fit(t)
            yhat_p            = self.ddiff(yhat)
            yhat_pp           = self.ddiff(yhat_p)
            noise             = self._get_noise(r)
            curvature         = (1 / self.ratio) * (1 / len(yhat_pp)) * np.sqrt(si.simps(yhat_pp**2))
            rng               = self.ratio * (np.max(self.averagedata[:, t]) - np.min(self.averagedata[:, t]))
            dmn               = self.ratio * len(self.averagedata[:, t])

            self.noises[t]    = np.log10(noise)
            self.curves[t]    = np.log10(curvature)
            self.ranges[t]    = np.log10(rng)
            self.domains[t]   = np.log10(dmn)
        return self.noises[t], self.curves[t], self.ranges[t], self.domains[t]

    def analyze_full(self):
        """
        Determine noise, curvature, range, and domain of specified data.
        Like analyze, except examines the entire file.

        :param: float->pixel to inch ratio

        :return: array->curvatures
        :return: array->noises
        :return: array->ranges
        :return: array->domains
        """
        if self.noises[0] == 0:
            s          = len(self.times)
            noises     = np.zeros(s)
            curvatures = np.zeros(s)
            rngs       = np.zeros(s)
            dmns       = np.zeros(s)
            for i in tqdm(range(s)):
                self.analyze(t=i)
        return self.noises, self.curves, self.ranges, self.domains


class ML(object):
    def __init__(self, args, algo='nn'):
        """
        Machine Learning to determine usability of data....
        """
        self.algo = get_algo(args, algo)

    def get_algo(self, args, algo):
        if algo == 'nn':
            return NearestNeighbor(args.nnk)

    def train(self):
        traindata = self.get_data()
        self.algo.train(traindata)

    def get_data(self):
        # We use the domain column to determine what fields have been filled out
        # If the domain is zero (i.e. not in error) than we should probably ignore it anyway
        traindata = data.get_traindata()
        return traindata

    def plot_fitspace(self, name, X, y, clf):
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        h = 0.01  # Mesh step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xlabel(r'$\log_{10}$ Noise')
        plt.ylabel(r'$\log_{10}$ Curvature')
        plt.savefig(name)


class NearestNeighbor(object):
    def __init__(self, k):
        """
        An example machine learning model. EVERY MODEL NEEDS TO PROVIDE:
            1. Train
            2. Predict
        """
        self.clf = neighbors.KNeighborsClassifier(k, weights='distance',
                                             p=2, algorithm='auto',
                                             n_jobs=8)

    def train(self, traindata):
        self.clf.fit(traindata[:, 1:5], traindata[:, 5])

    def predict(self, predictdata):
        return self.clf.predict(predictdata)


def get_args():
    """
    Get program arguments.

    Just use --help....
    """
    parser = argparse.ArgumentParser(prog='python3 linefit.py',
                description='Parameterize and analyze usability of conduit edge data')
    parser.add_argument('files', metavar='F', type=str, nargs='*',
                        help=('File(s) for processing. '
                            'Each file has a specific format: See README (or header) for specification.'))
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help=('Create Plot of file(s)? Note, unless --time flag used, will plot middle time.'))
    parser.add_argument('-pd', '--plotdata', action='store_true', default=False,
                        help=('Create plot of current datastore.'))
    parser.add_argument('-a', '--analyze', action='store_true', default=False,
                        help=('Analyze the file and determine Curvature/Noise parameters. '
                                'If --time not specified, will examine entire file. This will add results to '
                                'datastore with false flags in accept field if not provided.'))
    parser.add_argument('-mt', '--machinetest', action='store_true', default=False,
                        help=('Determine if the times from the file are usable based on supervised learning model. '
                                'If --time not specified, will examine entire file.'))
    parser.add_argument('-m', '--model', type=str, default='nn',
                        help=('Learning Model to use. Options are ["nn", "svm", "forest", "sgd"]'))
    parser.add_argument('-nnk', '--nnk', type=int, default=10,
                        help=('k-Parameter for k nearest neighbors. Google it.'))
    parser.add_argument('-t', '--time', type=int, default=None,
                        help=('Time (column) of data to use for analysis OR plotting. Zero-Indexed'))
    parser.add_argument('-d', '--datastore', type=str, default=DATASTORE,
                        help=("Datastore filename override. Don't do this unless you know what you're doing"))
    parser.add_argument('-pds', '--printdata', action='store_true', default=False,
                        help=("Print data"))
    parser.add_argument('-pdss', '--printdatashort', action='store_true', default=False,
                        help=("Print data short"))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())
