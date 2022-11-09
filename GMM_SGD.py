# -*- coding: utf-8 -*-
'''
Created on Tue Jul 27 17:52:58 2021

@author: Chiang-hua Tang
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def plot_contours(mu, sigma, title, pi=None, threshold=0.01, data_point=None,
    labels=None, label_coding=None):
    """
    Visualizing the clusters and the training data points.
    """
    if labels is None:
        labels = []
    if label_coding is None:
        label_coding = []

    plt.figure()
    # Plot the training data points if given.
    if data_point is not None:
        try:
            scatter = plt.scatter(x=data_point[:, 0], y=data_point[:, 1], s=2.5, c=labels)
            plt.legend(handles=scatter.legend_elements()[0], labels=label_coding)
        except Exception as e:
            print(str(e), end='\n\n')

    ##### Plot the Gaussians #####
    delta_x = 0.01  # resolution for the x-axis
    delta_y = 0.01  # resolution for the y-axis
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6

    # Pick the above-threshold clusters.
    if pi is not None:
        idx = pi > threshold
        mu = mu[idx]
        sigma = sigma[idx]

    k = mu.shape[0]
    for i in range(k):
        # Define the region of three sigmas.
        x_lower, x_upper = (np.clip(round(mu[i][0] - 3*(sigma[i][0, 0])**0.5, ndigits=3),
                            x_min, x_max),
                            np.clip(round(mu[i][0] + 3*(sigma[i][0, 0])**0.5, ndigits=3),
                            x_min, x_max))
        y_lower, y_upper = (np.clip(round(mu[i][1] - 3*(sigma[i][1, 1])**0.5, ndigits=3),
                            y_min, y_max),
                            np.clip(round(mu[i][1] + 3*(sigma[i][1, 1])**0.5, ndigits=3),
                            y_min, y_max))
        if x_lower == x_upper or y_lower == y_upper:
            break

        # Create xy coordinates.
        x = np.arange(x_lower, x_upper, delta_x)
        y = np.arange(y_lower, y_upper, delta_y)
        x_grid, y_grid = np.meshgrid(x, y)
        coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

        # Compute the probabilities.
        mean = mu[i]
        cov = sigma[i]
        try:
            z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        except Exception as e:
            print(str(e), end='\n\n')
            continue

        plt.contour(x_grid, y_grid, z_grid)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(title)
    plt.tight_layout()


def get_params(mu, sigma, idx=None, pi=None, print_out=False, prefix=''):
    """
    This function computes the parameters for the given mean vectors and covariance matrices (for 2D-Gaussian distributions).

    Parameters
    ----------
    mu : np.ndarray
        Mean vectors.

    sigma : np.ndarray
        Covariance matrices.

    idx : np.ndarray, optional
        An array which contains the indices to be computed. The default is None, where in this case all indices will be considered.

    pi : np.ndarray, optional
        Pi (weighting) vectors. The default is None.

    print_out : bool, optional
        If True, instead of returning a dictionary, the function prints out the parameters on the control panel. The default is False.

    Returns
    -------
    params : dict
        A dictionary which contains the values of the parameters (mu_x, mu_y, sd_x, sd_y, rho, pi).

    """
    if idx is None:
        idx = np.full(mu.shape[0], True)

    keys = ['mu_x', 'mu_y', 'sd_x', 'sd_y', 'rho']

    mu_x = mu[idx, 0]
    mu_y = mu[idx, 1]
    sd_x = (sigma[idx, 0, 0])**(1/2)
    sd_y = (sigma[idx, 1, 1])**(1/2)
    rho = sigma[idx, 0, 1] / (sd_x*sd_y)
    values = [mu_x, mu_y, sd_x, sd_y, rho]

    if pi is not None:
        keys.append('pi')
        values.append(pi[idx])

    params = {key: value for key, value in zip(keys, values)}

    if print_out:
        strings = ''
        for key, value in params.items():
            value_rounded = np.round(value, 3)
            strings += f'{prefix}{key} = {value_rounded}\n'
        print(strings)
    else:
        return params


def bhattacharyya_distance_gaussian(mu1, mu2, cov1, cov2):
    """
    Estimate the Bhattacharyya distance (between 2 Gaussian distributions).
    """
    cov = (1/2) * (cov1+cov2)

    T1 = (1/8) * ((mu1-mu2).T @ np.linalg.inv(cov) @ (mu1 - mu2))
    T2 = (1/2) * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))

    return T1 + T2


class GMM_SGD:
    def __init__(self, k=20,
        init_mu_means=np.array([0, 0], dtype=float),
        init_mu_covariances=np.array([[1.5**2, 0], [0, 1.5**2]], dtype=float),
        init_covariances=np.array([[0.5**2, 0], [0, 0.5**2]], dtype=float)):
        """
        GMM with SGD (based on Toscano & McMurray 2009).
        For the multi-dimensional (2D) model.

        Parameters
        ----------

        k : int, optional
            Number of initial distributions generated. The default is 20.

        init_mu_means : np.ndarray, optional
            Mean vector for the multivariate normal distribution from which the initial means are generated.

        init_mu_covariances : np.ndarray, optional
            Covariance matrix for the multivariate normal distribution from which the initial means are generated.

        init_covariances : np.ndarray, optional
            The initial covariance matrix.

        """
        self.k = k
        self.init_mu_means = init_mu_means
        self.init_mu_covariances = init_mu_covariances
        self.init_covariances = init_covariances

        # Initialze the distributions.
        self.generate_init_distributions()
        string = self.__repr__()
        print(string)

    def __repr__(self):
        params = get_params(self.mu, self.sigma, pi=self.pi)
        strings = f'There are {self.k} clusters.\n'
        for key, value in params.items():
            value_rounded = np.round(value, 3)
            strings += f'{key} = {value_rounded}\n'
        return strings

    def generate_init_distributions(self):
        """
        This medthod generates the initial distributions.
        """
        self.mu = np.random.multivariate_normal(self.init_mu_means, self.init_mu_covariances,
                                                self.k)
        self.sigma = np.full((self.k, 2, 2), self.init_covariances)
        self.pi = np.full(self.k, 1/self.k)

        plot_contours(self.mu, self.sigma, '0')

    def generate_training_distributions(self, df, training_pi=None):
        """
        This method generates the training distributions.

        Parameters
        ----------

        df : pd.DataFrame
            Real world dataset where the training distributions are generated from.

        training_pi : np.ndarray, optional
            An array which contains the weights for every training category. If not specified, the weights will be computed directly from the given real world dataset. The default is None.

        """
        df = df.copy()
        # Get the number of categories and the coding for each label.
        self.training_n = df.iloc[:, 0].nunique()
        self.training_label_coding = list(df.groupby(df.columns[0]).count().reset_index().iloc[:, 0])

        # Standardization
        df.iloc[:, 1:] = StandardScaler().fit_transform(df.iloc[:, 1:])

        # Reduce the features to 2 dimensions using PCA (for number of features > 2).
        features = str(list(df.columns[1:]))
        n_features = df.shape[1]-1
        if n_features > 2:
            pca = PCA(n_components=n_features)
            df.iloc[:, 1:] = pca.fit_transform(df.iloc[:, 1:])
            df = df.iloc[:, 0:3]  # keep only the first two components
            title = 'Real world dataset (PCA) ' + features
        else:
            title = 'Real world dataset ' + features

        # Fit the first two principle components to a 2D Gaussian and compute the parameters (pi, mu, sigma).
        if training_pi is None:  # regular cases
            #self.training_pi = df.groupby(df.columns[0]).count().iloc[:, 0].to_numpy(dtype=float)
            #self.training_pi = self.training_pi / np.sum(self.training_pi)  # Normalization
            self.training_pi = np.full(self.training_n, 1/self.training_n)
            self.NH_flag = False
        else:  # the NH evolutional case
            self.training_pi = training_pi
            self.NH_flag = True

        self.training_mu = df.groupby(df.columns[0]).mean().to_numpy(dtype=float)
        self.training_sigma = df.groupby(df.columns[0]).cov().to_numpy(dtype=float).reshape(self.training_n, 2, 2)

        # Print out the training parameters.
        print(f'There are {self.training_n} training categories: {self.training_label_coding}')
        params = get_params(self.training_mu, self.training_sigma, pi=self.training_pi)
        strings = ''
        for key, value in params.items():
            value_rounded = np.round(value, 3)
            strings += f'{key} = {value_rounded}\n'
        print(strings)

        # Visualization
        data_point = df.iloc[:, np.r_[1, 2]].to_numpy()
        labels, label_coding = pd.factorize(df.iloc[:, 0], sort=True)
        label_coding = list(label_coding)
        plot_contours(self.training_mu, self.training_sigma, title,
            data_point=data_point, labels=labels, label_coding=label_coding)

    def _SGD_with_competition(self, training_data_set, skip_low_pi, lr_pi, lr_mu, lr_sd, lr_rho,
        skip_threshold):
        """
        This method serves as the main algorithm for the siumulation. It sets up the initial distributions and updates the parameters using stochastic gradient descent (with competition).
        """
        # Run through the training samples.
        for i, data_point in tqdm(enumerate(training_data_set, start=1)):
            p_list = []
            overall_likelihood = 0

            # Compute the likelihoods for each Gaussian.
            break_out_flag = False
            for j in range(k):
                try:
                    p = self.pi[j] * multivariate_normal(self.mu[j], self.sigma[j]).pdf(data_point)
                    p_list.append(p)
                    overall_likelihood += p
                except:  # det(cov) = 0 or inf
                    break_out_flag = True
                    break
            if break_out_flag:
                break

            # Find the distribution that has the maximum p and update pi 'only' for this particular distribution (Competition!).
            max_idx = p_list.index(max(p_list))
            G = self.pi[max_idx] * multivariate_normal(self.mu[max_idx], self.sigma[max_idx]).pdf(data_point)
            M = overall_likelihood

            # Update pi and normalize.
            pi = self.pi[max_idx]
            self.pi[max_idx] = pi + (G/pi)*lr_pi/M  # update
            self.pi = self.pi / np.sum(self.pi)  # normalize

            # Get the parameters.
            if skip_low_pi:  # get the above-skip_threshold indices
                idx = self.pi > skip_threshold
            else:  # get all indices
                idx = np.full(k, True)

            params = get_params(mu=self.mu, sigma=self.sigma, idx=idx)
            mu_x = params['mu_x']
            mu_y = params['mu_y']
            sd_x = params['sd_x']
            sd_y = params['sd_y']
            rho = params['rho']
            x = data_point[0]
            y = data_point[1]
            G = [self.pi[i] * multivariate_normal(self.mu[i], self.sigma[i]).pdf(data_point)
                for i in np.where(idx)[0]]

            # Define the derivatives for mu, sd, and rho.
            d_mu_x = G * (1/(1-rho**2)) * ((x-mu_x) / (sd_x**2) + (rho/(sd_x*sd_y))*(y-mu_y))
            d_mu_y = G * (1/(1-rho**2)) * ((y-mu_y) / (sd_y**2) + (rho/(sd_y*sd_x))*(x-mu_x))
            d_sd_x = G * ((-1/sd_x) + ((((x-mu_x)**2)/(sd_x)**3) -
                (rho*(x-mu_x)*(y-mu_y)/((sd_x**2)*sd_y)))/(1-rho**2))
            d_sd_y = G * ((-1/sd_y) + ((((y-mu_y)**2)/(sd_y)**3) -
                (rho*(y-mu_y)*(x-mu_x)/((sd_y**2)*sd_x)))/(1-rho**2))
            d_rho = G * (1/(1-rho**2)) * (rho - (1/(1-rho**2)) * (rho * (((x-mu_x)/sd_x)**2 +
                ((y-mu_y)/sd_y)**2) - ((rho**2+1)*((x-mu_x)/sd_x)*((y-mu_y)/sd_y))))

            # Update mu.
            self.mu[idx, 0] = mu_x + d_mu_x*lr_mu/M
            self.mu[idx, 1] = mu_y + d_mu_y*lr_mu/M
            # Update sd and rho.
            sd_x_new = sd_x + d_sd_x*lr_sd/M
            sd_y_new = sd_y + d_sd_y*lr_sd/M
            self.sigma[idx, 0, 0] = sd_x_new**2
            self.sigma[idx, 1, 1] = sd_y_new**2
            self.sigma[idx, 0, 1] = self.sigma[idx, 1, 0] = (rho + d_rho*lr_rho/M)*sd_x_new*sd_y_new

            if i in [100, 1000, 10000, 20000, 50000, 100000, 150000, 200000]:
                plot_contours(self.mu, self.sigma, title=str(i), pi=self.pi)

        if break_out_flag: # ERROR!
            print(f'Error for repetition = {i} and index = {j}, det = {np.linalg.det(self.sigma[j])}.', end='\n\n')
            print(self.pi[j], self.mu[j], self.sigma[j], sep='\n\n', end='\n\n')
            return False
        else:
            return True

    def _test(self, testing_pi, testing_mu, testing_sigma, pi, mu, sigma, n_data=1000):
        """
        This method tests the performance of the model by fitting a testing dataset to the model and returns a confusion matrix.
        """
        # Initialize the confusion matrix.
        n = self.training_n
        cm = np.zeros((n, n), dtype=int)

        for i in range(n_data):
            # Generate testing data points.
            testing_label = np.random.choice(n, size=1, p=testing_pi)[0]
            testing_data_point = np.random.multivariate_normal(testing_mu[testing_label],
                testing_sigma[testing_label])

            # Compute the posterior probability for each cluster.
            p_list = []
            for j in range(n):
                p = pi[j] * multivariate_normal(mu[j], sigma[j]).pdf(testing_data_point)
                p_list.append(p)

            # Find the index that has maximum p and adjust the confusion matrix.
            predicting_label = p_list.index(max(p_list))
            cm[predicting_label, testing_label] += 1

        # Compute the precision rates.
        pr = (np.diag(cm) / np.sum(cm, axis=0)).round(3) * 100
        # Convert to dataframe.
        cm = pd.DataFrame(cm, columns=self.training_label_coding, index=self.training_label_coding)
        print(cm, pr, sep='\n', end='\n\n')

    def sim(self, n_generation: int, n_data: int, skip_low_pi=False,
        lr_pi=0.001, lr_mu=0.01, lr_sd=0.01, lr_rho=0.01, threshold=0.1, skip_threshold=0.01):
        """
        This method runs the simulation for 1 or more generations. For cases with more than 1 generation, the resulting distributions of each generation will be taken to generate the training dataset for the next generation (and so on).

        Parameters
        ----------

        n_generation : int
            Number of generations.

        n_data : int
            Number of training data points (epochs) generated per generation.

        skip_low_pi : bool, optional
            If true, skip the distributions with pi < skip_threshold (for time-conserving). The default is False.

        lr_pi : float, optional
            Learning rate for pi. The default is 0.001.

        lr_mu : float, optional
            Learning rate for mu. The default is 0.01.

        lr_sd : float, optional
            Learning rate for sd. The default is 0.01.

        lr_rho : float, optional
            Learning rate for rho. The default is 0.01.

        threshold : float, optional
            Information for distributions which have their weights above this value will be shown. The default is 0.01.

        skip_threshold : float, optional
            Threshold value for skipping distributions with low pi values. The default is 0.01.

        """
        # Preserve the training parameters of gen 1.
        training_pi_1 = self.training_pi
        training_mu_1 = self.training_mu
        training_sigma_1 = self.training_sigma
        # Plot the training distributions for gen 1.
        plot_contours(training_mu_1, training_sigma_1, title='training')
        print(self.training_label_coding, end='\n\n')

        # Run through n_generation gens.
        for gen in range(n_generation):
            print(f'Generation {gen+1}:')
            if gen > 1:
                self.generate_init_distributions()

            # Generate the training dataset.
            generators = [multivariate_normal(self.training_mu[i], self.training_sigma[i]) for i in
                range(self.training_n)]
            labels = np.random.choice(self.training_n, size=n_data, p=self.training_pi)
            training_data_set = np.array([generators[label].rvs() for label in labels])

            ##### Main part of the simulation #####
            # No error occurs.
            if self._SGD_with_competition(training_data_set, skip_low_pi, lr_pi, lr_mu, lr_sd,
                lr_rho, skip_threshold):
                if self.NH_flag:
                    # Pick the clusters with the largest pi values.
                    idx = self.pi.argsort()[-1:-self.training_n-1:-1]
                    resulting_pi = self.pi[idx]
                    resulting_mu = self.mu[idx]
                    resulting_sigma = self.sigma[idx]
                else:
                    # Pick the clusters with pi > threshold.
                    idx = self.pi > threshold
                    resulting_pi = self.pi[idx]
                    resulting_mu = self.mu[idx]
                    resulting_sigma = self.sigma[idx]
                    n_above_threshold_clusters = idx.sum()

                plot_contours(mu=resulting_mu, sigma=resulting_sigma, title=gen+1,
                    pi=resulting_pi, data_point=training_data_set, labels=labels,
                    label_coding=self.training_label_coding)

                # Failed!
                if (not self.NH_flag) and (n_above_threshold_clusters != self.training_n):
                    print(f'Failed, resulting in {n_above_threshold_clusters} clusters.')
                    get_params(mu=resulting_mu, sigma=resulting_sigma,
                        pi=resulting_pi, print_out=True)
                    print(f'Simulation stopped at generation {gen+1}.')
                    break
                # Success!
                else:
                    print('Success!')
                    # Sort the clusters to match each category (based on the Euclidian distances between resulting means and training means).
                    idx_array = np.array(list(permutations(range(self.training_n))))
                    if self.NH_flag:
                        # Define "xi" as the cluster with the smallest pi value.
                        resulting_xi_idx = resulting_pi.argmin()
                        training_xi_idx = self.training_pi.argmin()
                        idx_array = idx_array[idx_array[:,training_xi_idx] == resulting_xi_idx]
                    dists = cdist(resulting_mu, self.training_mu)
                    sums = dists[idx_array, list(range(self.training_n))].sum(axis=1)
                    sorted_idx = idx_array[sums.argmin()]
                    resulting_pi = resulting_pi[sorted_idx]
                    resulting_mu = resulting_mu[sorted_idx]
                    resulting_sigma = resulting_sigma[sorted_idx]

                    # Get the parameters and print out the results.
                    resulting_pi = resulting_pi / np.sum(resulting_pi)
                    get_params(mu=self.training_mu, sigma=self.training_sigma,
                        pi=self.training_pi, print_out=True, prefix='training_')
                    get_params(mu=resulting_mu, sigma=resulting_sigma,
                        pi=resulting_pi, print_out=True, prefix='resulting_')

                    # Calculate the training and resulting Bhattacharyya distances.
                    training_bds = {}
                    resulting_bds = {}
                    for idx in combinations(range(self.training_n), 2):
                        cat1 = self.training_label_coding[idx[0]]
                        cat2 = self.training_label_coding[idx[1]]

                        mu1 = self.training_mu[idx[0]]
                        mu2 = self.training_mu[idx[1]]
                        cov1 = self.training_sigma[idx[0]]
                        cov2 = self.training_sigma[idx[1]]
                        bd = bhattacharyya_distance_gaussian(mu1, mu2, cov1, cov2)
                        training_bds[f'{cat1}-{cat2}'] = np.round(bd, 3)

                        mu1 = resulting_mu[idx[0]]
                        mu2 = resulting_mu[idx[1]]
                        cov1 = resulting_sigma[idx[0]]
                        cov2 = resulting_sigma[idx[1]]
                        bd = bhattacharyya_distance_gaussian(mu1, mu2, cov1, cov2)
                        resulting_bds[f'{cat1}-{cat2}'] = np.round(bd, 3)

                    print(f'Training Bhattacharyya distances: {training_bds}')
                    print(f'Mean training Bhattacharyya distance: '
                        f'{np.round(np.mean(list(training_bds.values())), 3)}')
                    print(f'Resulting Bhattacharyya distances: {resulting_bds}')
                    print(f'Mean resulting Bhattacharyya distance: '
                        f'{np.round(np.mean(list(resulting_bds.values())), 3)}', end='\n\n')

                    if gen+1 < n_generation:
                        # Take the previous resulting distributions (mu, sigma) as the training distributions for the next generation.
                        self.training_mu = resulting_mu
                        self.training_sigma = resulting_sigma
            # Error occurs!
            else:
                print(f'Simulation stopped at generation {gen+1}.')
                plot_contours(mu=self.mu, sigma=self.sigma, title="Error", pi=self.pi,
                    data_point=training_data_set, labels=labels,
                    label_coding=self.training_label_coding)
                break
