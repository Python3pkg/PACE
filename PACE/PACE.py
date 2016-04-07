#!/usr/bin/env python3.5


"""
PACE

TODO:
    * model training/testing
        * more models (technically)
    * multithreading

"""


import sys
import os
import argparse
import hashlib
import typing

from enforce import runtime_validation as types

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
            fit.plot_file(name=plot_name, time=args.time)
    if args.analyze:
        for filename in args.files:
            manage_file_analysis(args, filename, data)
    if args.plotdata:
        data.plot_traindata()
    if args.machinetest:
        learner = ML(algo=args.model)
    if args.printdata:
        data.printdata()
    if args.printdatashort:
        data.printshort()


@types
def manage_file_analysis(args: argparse.Namespace, filename: str, data: object) -> None:
    """
    Take care of the analysis of a datafile
    """
    key = DataStore.hashfile(filename)
    print('Analyzing {} --> {}'.format(filename, key))
    if data.check_key(key):  # if exists in database, prepopulate
        fit = LineFit(filename, data=data.get_data(key))
    else:
        fit = LineFit(filename)
    if args.time:
        noise, curvature, rnge, domn = fit.analyze(time=args.time)
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
    def __init__(self, name: str):
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

        :param: name of datastore
        """
        self.name = name
        self.data = {}

    def load(self) -> None:
        """
        Load datafile
        """
        try:
            self.data = sco.loadmat(self.name)
        except FileNotFoundError:
            pass

    def save(self) -> None:
        """
        Save datafile to disk
        """
        sco.savemat(self.name, self.data)

    def get_data(self, key: str) -> np.ndarray:
        """
        Returns the specified data. Warning, ZERO ERROR HANDLING

        :param key: name of file

        :return: 2d data array
        """
        return self.data[key]

    @types
    def get_keys(self) -> typing.List[str]:
        """
        Return list of SHA512 hash keys that exist in datafile

        :return: list of keys
        """
        keys = []
        for key in self.data.keys():
            if key not in ['__header__', '__version__', '__globals__']:
                keys.append(key)
        return keys

    @types
    def check_key(self, key: str) -> bool:
        """
        Checks if key exists in datastore. True if yes, False if no.

        :param: SHA512 hash key

        :return: whether or key not exists in datastore
        """
        keys = self.get_keys()
        return key in keys

    def get_traindata(self) -> np.ndarray:
        """
        Pulls all available data and concatenates for model training

        :return: 2d array of points
        """
        traindata = None
        for key, value in self.data.items():
            if key not in ['__header__', '__version__', '__globals__']:
                if traindata is None:
                    traindata = value[np.where(value[:, 4] != 0)]
                else:
                    traindata = np.concatenate((traindata, value[np.where(value[:, 4] != 0)]))
        return traindata

    @types
    def plot_traindata(self, name: str='dataplot') -> None:
        """
        Plots traindata.... choo choo...
        """
        traindata = self.get_traindata()

        plt.figure(figsize=(16, 16))
        plt.scatter(traindata[:, 1], traindata[:, 2],
                    c=traindata[:, 5], marker='o', label='Datastore Points')
        plt.xlabel(r'$\log_{10}$ Noise')
        plt.ylabel(r'$\log_{10}$ Curvature')
        plt.legend(loc=2, fontsize='xx-large')
        plt.savefig('./img/{}.png'.format(name))

    def printdata(self) -> None:
        """ Prints data to stdout """
        np.set_printoptions(threshold=np.nan)
        print(self.data)
        np.set_printoptions(threshold=1000)

    def printshort(self) -> None:
        """ Print shortened version of data to stdout"""
        print(self.data)

    @types
    def update(self, key: str, data: np.ndarray) -> None:
        """ Update entry in datastore """
        self.data[key] = data

    def update1(self, key: str, data: np.ndarray, size: int) -> None:
        """ Update one entry in specific record in datastore """
        print(data)
        if key in self.get_keys():
            self.data[key][data[0]] = data
        else:
            newdata = np.zeros((size, 6))
            newdata[data[0]] = data
            self.data[key] = newdata

    @staticmethod
    @types
    def hashfile(name: str) -> str:
        """
        Gets a hash of a file using block parsing

        http://stackoverflow.com/questions/3431825/generating-a-md5-checksum-of-a-file
        Using SHA512 for long-term support (hehehehe)
        """
        hasher = hashlib.sha512()
        with open(name, 'rb') as openfile:
            for chunk in iter(lambda: openfile.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


class LineFit(object):
    def __init__(self, filename: str, data: np.ndarray=None,
                 function_number: int=16, spread_number: int=22):
        """
        Main class for line fitting and parameter determination

        :param: filename
        :param: data for fitting
        :param: number of functions
        :param: gaussian spread number
        """
        self.filename = filename
        (self.averagedata, self.times,
         self.accepts, self.ratio, self.viscosity) = self._loadedges()
        self.domain = np.arange(len(self.averagedata[:, 0]))
        self.function_number = function_number
        self.spread_number = spread_number
        if data is None:
            self.noises = np.zeros(len(self.times))
            self.curves = np.zeros(len(self.times))
            self.ranges = np.zeros(len(self.times))
            self.domains = np.zeros(len(self.times))
        else:
            self.noises = data[:, 1]
            self.curves = data[:, 2]
            self.ranges = data[:, 3]
            self.domains = data[:, 4]

    @types
    def _loadedges(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Attempts to intelligently load the .mat file and take average of left and right edges

        :return: left and right averages
        :return: times for each column
        :return: accept/reject for each column
        :return: pixel-inch ratio
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
            ratio = data['ratio']
        except KeyError:
            ratio = 1

        try:
            viscosity = data['viscosity']
        except KeyError:
            viscosity = np.ones(len(times))
        return averagedata, times, accept, ratio, viscosity

    def plot_file(self, name: str=None, time: int=None) -> None:
        """
        Plot specific time for provided datafile.
        If no time provided, will plot middle.

        :param: savefile name
        :param: time/data column
        """
        if not time:
            time = int(len(self.times) / 2)
        if not name:
            name = './img/' + self.filename + '.png'
        yhat, residuals, residual_mean, noise = self._get_fit(time)
        plt.figure()
        plt.scatter(self.domain, self.averagedata[:, time], alpha=0.2)
        plt.plot(yhat)
        plt.savefig(name)

    @staticmethod
    @types
    def ddiff(arr: np.ndarray) -> np.ndarray:
        """
        Helper Function: Divided Differences

        input: array
        """
        return arr[:-1] - arr[1:]

    @types
    def _gaussian_function(self, datalength: int, values: np.ndarray,
                           height: int, index: int) -> np.ndarray:
        """
        i'th Regression Model Gaussian

        :param: len(x)
        :param: x values
        :param: height of gaussian
        :param: position of gaussian

        :return: gaussian bumps over domain
        """
        return height * np.exp(-(1 / (self.spread_number * datalength)) *
                               (values - ((datalength / self.function_number) * index)) ** 2)

    @types
    def _get_fit(self, time: int) -> typing.Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Fit regression model to data

        :param: time (column of data)

        :return: predicted points
        :return: residuals
        :return: mean residual
        :return: error
        """
        rawdata = self.averagedata[:, time]
        domain = np.arange(len(rawdata))
        datalength = len(domain)
        coefficients = np.zeros((datalength, self.function_number + 2))
        coefficients[:, 0] = 1
        coefficients[:, 1] = domain
        for i in range(self.function_number):
            coefficients[:, 2 + i] = self._gaussian_function(datalength, domain, 1, i)
        betas = linalg.inv(coefficients.transpose().dot(coefficients)).dot(coefficients.transpose().dot(rawdata))
        predicted_values = coefficients.dot(betas)
        residuals = rawdata - predicted_values
        error = np.sqrt(residuals.transpose().dot(residuals) / (datalength - (self.function_number + 2)))
        return predicted_values, residuals, residuals.mean(), error

    @types
    def _get_noise(self, residuals: np.ndarray) -> float:
        """
        Determine Noise of Residuals.

        :param: residuals

        :return: noise
        """
        return np.mean(np.abs(residuals))

    @types
    def analyze(self, time: int=None) -> typing.Tuple[float, float, int, int]:
        """
        Determine noise, curvature, range, and domain of specified array.

        :param: pixel to inch ratio
        :param: time (column) to use.

        :return: curvature
        :return: noise
        :return: range
        :return: domain
        """
        if not time:
            time = int(len(self.times) / 2)
        if self.domains[time] == 0:
            yhat, residuals, mean_residual, error = self._get_fit(time)
            yhat_p = self.ddiff(yhat)
            yhat_pp = self.ddiff(yhat_p)
            noise = self._get_noise(residuals)
            curvature = (1 / self.ratio) * (1 / len(yhat_pp)) * np.sqrt(si.simps(yhat_pp ** 2))
            rng = (self.ratio * (np.max(self.averagedata[:, time]) -
                                 np.min(self.averagedata[:, time])))
            dmn = self.ratio * len(self.averagedata[:, time])

            self.noises[time] = np.log10(noise)
            self.curves[time] = np.log10(curvature)
            self.ranges[time] = np.log10(rng)
            self.domains[time] = np.log10(dmn)
        return self.noises[time], self.curves[time], self.ranges[time], self.domains[time]

    @types
    def analyze_full(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            timelength = len(self.times)
            for i in tqdm(range(timelength)):
                self.analyze(time=i)
        return self.noises, self.curves, self.ranges, self.domains


class ML(object):
    def __init__(self, args: argparse.Namespace, algo: str='nn'):
        """
        Machine Learning to determine usability of data....
        """
        self.algo = self.get_algo(args, algo)

    def get_algo(self, args: argparse.Namespce, algo: str) -> object:
        """ Returns machine learning algorithm based on arguments """
        if algo == 'nn':
            return NearestNeighbor(args.nnk)

    def train(self) -> None:
        """ Trains specified algorithm """
        traindata = self.get_data()
        self.algo.train(traindata)

    def get_data(self) -> np.ndarray:
        """
        Gets data for training

        We use the domain column to determine what fields have been filled out
        If the domain is zero (i.e. not in error) than we should probably ignore it anyway
        """
        traindata = data.get_traindata()
        return traindata

    def plot_fitspace(self, name: str, X: np.ndarray, y: np.ndarray, clf: object) -> None:
        """ Plot 2dplane of fitspace """
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
    def __init__(self, k: int):
        """
        An example machine learning model. EVERY MODEL NEEDS TO PROVIDE:
            1. Train
            2. Predict
        """
        self.clf = neighbors.KNeighborsClassifier(k, weights='distance',
                                                  p=2, algorithm='auto',
                                                  n_jobs=8)

    def train(self, traindata: np.ndarray) -> None:
        """ Trains on dataset """
        self.clf.fit(traindata[:, 1:5], traindata[:, 5])

    def predict(self, predictdata: np.ndarray) -> np.ndarray:
        """ predict given points """
        return self.clf.predict(predictdata)


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


if __name__ == '__main__':
    sys.exit(main())
