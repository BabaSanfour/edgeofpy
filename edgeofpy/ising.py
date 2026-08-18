"""Phase diagram analysis
"""

import multiprocessing as mp
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt


def fit_pmem(data, iter_max : int = 5000000, error_max=1e-8, eta=.05, 
             method='donders', init_zero=False, verbose=False):
    """Fit binary multichannel data to the pairwise maximum entropy model by
    pseudo-likelihood maximization.    

    Parameters
    ----------
    binarized_data : 2d array (dtype='float')
        Multi-channel binary data, of shape `n_node` x `dat_len`.
    iter_max : int
        Maximum number of iterations of the gradient ascent. 
    error_max : float
        Maximum tolerated error, exclusive. 'ezaki' default: 1e-8
    eta : float
        Learning rate.
    method : str
        PLM algorithm. 'donders' or 'ezaki'
    init_zero : bool
        If True, initialize h and J as zeros; if False, initialize as 
        data mean and data correlation matrix, respectively.
    verbose : bool
        If True, iteration and error will be printed at every step.

    Returns
    -------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field, i.e. the intrinsic 
        likelihood that S_i = 1.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.

    References
    ----------
    Ezaki et al. (2020) Nat Comm 3(1), 1-9.
    Tkačik et al. (2015) PNAS 112(37), 11508–13.
    
    Acknowledgements
    ----------------
    This function was partly adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    The `'donders'` method was adapted from Matlab code by Hilbert 
    Kappen.
    """
    n_node, dat_len = data.shape
    dat_mean = data.mean(1, keepdims=True)
    dat_corr = data @ data.T / dat_len
    
    if init_zero:
        h = np.zeros((n_node, 1))
        J = np.zeros((n_node, n_node))
    else:
        h = dat_mean.copy()
        J = dat_corr.copy() - dat_mean ** 2
        np.fill_diagonal(J, 0)
    
    if method == 'ezaki':
        for t in range(iter_max):
            if verbose:
                print(f'Iteration {t}')
            h, J, error = _iter_fit_pmem(data, h, J, n_node, dat_len, dat_mean, 
                                    dat_corr, eta)
            if verbose:
                print(f'Error = {error}')
            if error < error_max:
                print(f'Converged at iteration {t}')
                break
        h = h.squeeze()
    elif method == 'donders':
        L = 1e10
        t = 0
        while L > error_max:
            print(f'iteration {t-1}')
            t += 1
            sigma = np.tanh(J @ data + h[:, None])
            print(sigma.shape)
            C = sigma @ data.T / dat_len
            C = 0.5 * (C + C.T)
            dJ = dat_corr - C
            np.fill_diagonal(dJ, 0)
            dh = dat_mean - np.sum(sigma, axis=1) / dat_len
            J = J + eta * dJ
            h = h + eta * dh
            L1 = np.sum(dJ**2) + np.sum(dh**2)
            if t % 100 == 0:
                print(f't = {t-1}, L1: {L1}')
            if L1 > L:
                eta = eta / 2
                print(f't = {t-1}, eta = {eta}, L1 = {L1}')
            L = L1

        print(f'Converged at iteration {t-1}')
    else:
        raise ValueError('Invalid method.')
    
    return h, J

def _iter_fit_pmem(data, h, J, n_node, dat_len, dat_mean, dat_corr, eta):
    """Iterated part of pairwise maximum entropy model fitting"""
    prediction = np.tanh(J @ data + h)
    Ldh = -dat_mean + np.mean(prediction, 1, keepdims=True)
    h = h - Ldh * eta
    
    corr_mat = data @ prediction.T / dat_len # hybrid
    LdJ = (-dat_corr + 0.5 * corr_mat + 0.5 * corr_mat.T)
    np.fill_diagonal(LdJ, 0) # eliminate diagonal elements (J_{ii} = 0)
    J = J - LdJ * eta
    
    error = ( np.sqrt( np.linalg.norm(LdJ, 'fro')**2 + np.linalg.norm(Ldh)**2 ) 
                / n_node / (n_node + 1) )

    return h, J, error

def fit_beta(data, h, J, iter_max : int = 5000000, error_max=0.00000001, 
             eta=.005, verbose=False):
    """Fine-tune a PMEM fit using temperature.  

    Parameters
    ----------
    binarized_data : 2d array (dtype='float')
        Multi-channel binary data, of shape `n_node` x `dat_len`.
    h : 1d array (dtype='float')
        The elementwise intrinsic field, i.e. the intrinsic 
        likelihood that S_i = 1.
    J : 2d array (dtype='float')
        The n_node x n_node interaction matrix.
    iter_max : int
        Maximum number of iterations of the gradient ascent. 
    error_max : float
        Maximum tolerated error, exclusive. 'ezaki' default: 1e-8
    eta : float
        Learning rate.
    verbose : bool
        If True, iteration and error will be printed at every step.

    Returns
    -------
    beta : float
        The estimated inverse temperature of the fine-tuned model.

    References
    ----------
    Ruffini et al. (2023) PLOS Comp Biol 19(2), e1010811.
    """
    n_node, dat_len = data.shape
    if h.ndim == 1:
        h = h[:, None]
    beta = 1.0

    H = np.zeros(dat_len)
    for t in range(dat_len):
        sigma = data[:, t]#[:, None]
        H[t] = ( np.inner(-0.5 * np.inner(J, sigma), sigma) 
                - np.inner(h.squeeze(), sigma) )
    H_mean = np.mean(H)
    print(f'empirical H: {H_mean}')

    for t in range(iter_max):
        if verbose:
            print(f'Iteration {t}')
        beta, error = _iter_fit_beta(data, beta, h, J, H_mean, n_node, 
                                  dat_len, eta)
        if verbose:
            print(f'Beta = {beta}, Error = {error}')
        if error < error_max:
            print(f'Converged at iteration {t}')
            break
    
    return beta

def _iter_fit_beta(data, beta, h, J, H, n_node, dat_len, eta):
    """Iterated part of the temperature fitting"""
    # get model prediction
    prediction = np.tanh( (J * beta) @ data + h * beta)

    # get 1- and 2-point correlations
    one_pt_corr = np.sum(prediction, 1, keepdims=True) / dat_len
    two_pt_corr = data @ prediction.T / dat_len # hybrid

    # make symmetrical
    two_pt_corr = 0.5 * (two_pt_corr + two_pt_corr.T)

    # eliminate diagonal elements
    np.fill_diagonal(two_pt_corr, 0)
    np.fill_diagonal(J, 0)

    model_mean_H = -np.sum(h * one_pt_corr) - 0.5 * np.sum(J * two_pt_corr)
    Ldbeta = H - model_mean_H
    beta = beta - Ldbeta * eta

    error = np.abs(Ldbeta) / (n_node * (n_node + 1))

    return beta, error

def metropolis(h, J, beta=1.0, n_iter=1000, rand_seed=0, rand_init=False, 
               save_every=None, save_path=None, checkpoint_file=None):
    """Sample data from the estimated model using the Metropolis-Hastings 
    algorithm and compute statistics.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field, i.e. the intrinsic 
        likelihood that S_i = 1.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.
    beta : float, optional
        The inverse temperature.
    n_iter : int
        Number of iterations.
    rand_seed : int
        Seed for random number generator.
    rand_init : bool
        If True, the initial spin configuration `sigma` will be randomly
        generated using rand_seed. If False, `sigma` will be all ones.
    save_every : int, optional
        If not None, save the simulation every `save_every` iterations.
    save_path : str, optional
        Path to save the simulation.
    checkpoint_file : str, optional
        If a path is specified, the simulation will continue from this
        checkpoint.

    Returns
    -------
    m : float
        Magnetization order parameter.
    q : float
        Spin-glass order parameter.
    chi_u : float
        Uniform susceptibility.
    chi_sg : float
        Spin-glass susceptiblity.
    C : float
        Specific heat.
    local_m : 2d array, dtype='float'
        Local mean.
    corr : 2d array, dtype='float'
        Correlation matrix.

    References
    ----------
    Ezaki et al. (2020) Nat Comm 3(1), 1-9.
    Fischer & Hertz (1991) "Spin glasses". Cambridge UP.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    if save_every is not None and save_path is None:
        raise ValueError("Specify a path for saving checkpoints.")
    
    np.random.seed(rand_seed)
    n_node = len(h)

    if checkpoint_file is None:
        local_m = np.zeros(n_node)
        corr = np.zeros((n_node, n_node))
        if rand_init:
            sigma = np.sign( np.random.rand(n_node) - 0.5 )
        else:
            sigma = np.ones(n_node) # all nodes start "on"
        n_step = n_iter * n_node
        E = 0
        E2 = 0        
    
        # First, run out the transient for n_step steps
        for t in range(n_step):
            n = np.random.randint(low=0, high=n_node)
            dE = -2 * h[n] * sigma[n] - 2 * np.inner(J[:, n], sigma) * sigma[n]
            p = min(1, np.exp(dE * beta))
            if np.random.random() < p:
                sigma[n] = -sigma[n]

    else:
        checkpoint = np.load(checkpoint_file)
        h = checkpoint['h']
        J = checkpoint['J']
        sigma = checkpoint['sigma']
        local_m = checkpoint['local_m']
        corr = checkpoint['corr']
        E = checkpoint['E']
        E2 = checkpoint['E2']
        n_step = checkpoint['n_step_remaining']
  
    # Now, sample the steady-state every n_node steps
    for t in range(n_step):
        n = np.random.randint(low=0, high=n_node)
        dE = -2 * h[n] * sigma[n] - 2 * np.inner(J[:, n], sigma) * sigma[n]
        p = min(1, np.exp(dE * beta))
        if np.random.random() < p:
            sigma[n] = -sigma[n]
            
        # Local mean and correlation matrix are updated every `n_node` steps.
        if t % n_node == 0:
            local_m = local_m + sigma
            corr = corr + np.outer(sigma, sigma) # two minuses cancel out
            energy = ( np.inner(-0.5 * np.inner(J, sigma), sigma) 
                      - np.inner(h, sigma) )
            E = E - energy
            E2 = E2 + energy**2

            # Save
            if save_every is not None and t % (n_node * save_every) == 0:
                n_step_remaining = n_step - t
                np.savez(f'{save_path}_{int(t / n_node)}.npz', h=h, J=J,
                         sigma=sigma, local_m=local_m, corr=corr, E=E, E2=E2,
                         n_step_remaining=n_step_remaining)
        
    local_m = local_m / n_iter
    corr = corr / n_iter
    covariance = corr - np.outer(local_m, local_m)
    E = E / n_iter
    E2 = E2 / n_iter
    C = beta**2 * (E2 - E**2) / n_node
    q = np.inner(local_m, local_m) / n_node
    m = np.sum(local_m) / n_node
    eigenspectrum = np.linalg.eig(covariance)[0]
    chi_sg = beta**2 * np.inner(eigenspectrum, eigenspectrum) / n_node
    chi_u = beta * np.sum(covariance) / n_node
    
    return m, q, chi_u, chi_sg, C, local_m, corr

def sk_diagram(h, J, beta=1.0, mu_range=(-0.002, .0095), mu_step=.0005, 
               std_range=(.005, .1), std_step=.005, mu_width=None, 
               std_width=None, n_iter=1000, rand_seed=0, rand_init=False,
               multiproc=False, n_cpu=None, save_every=None,
               save_path=None):
    """Create a Sherrington-Kirkpatrick phase diagram by fixing the estimated
    intrinsic field h and sweeping through different values of the mean and 
    standard deviation of the estimated interaction matrix J. The x-axis is 
    the mean and the y-axis is the standard deviation.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.    
    beta : float, optional
        The inverse temperature. Default at unit temperature 1.0
    mu_range : tuple of float
        Minimum and maximum values for the parameter sweep of the mean.
    mu_step : float
        Step for the parameter sweep of the mean.
    std_range : tuple of float
        Minimum and maximum values for the parameter sweep of the standard 
        deviation.
    std_step : float
        Step for the parameter sweep of the standard deviation.
    mu_width : float, optional
        Replaces `mu_range`. `mu_width` then determines the interval to plot, 
        centered around the mean of J.
    std_width : float, optional
        Replaces `std_range`. `std_width` then determines the interval to plot, 
        centered around the standard deviation of J.
    n_iter : int
        Number of iterations for Metropolis sampling.
    rand_seed : int
        Seed for random number generator.
    rand_init : bool
        If True, the initial spin configuration `sigma` for each simulation
        will be randomly generated using rand_seed. If False, `sigma` will be 
        all ones.
    multiproc : bool
        Whether to use multiprocessing.
    n_cpu : int or None
        For use with multiprocessing. If None, will use mp.cpu_count().
    save_every : int, optional
        If not None, save the simulation every `save_every` iterations.
    save_path : str, optional
        Path to save the simulation.

    Returns
    -------
    m : 2d array
        Phase diagram in terms of the magnetization order parameter.
    q : 2d array
        Phase diagram in terms of the spin-glass order parameter..
    chi_u : 2d array
        Phase diagram in terms of the uniform susceptibility.
    chi_sg : 2d array
        Phase diagram in terms of the spin-glass susceptibility.
    C : 2d array
        Phase diagram in terms of the specific heat.
    x_range : 1d array
        Values for the mean.
    y_range : 1d array
        Values for the standard deviation.

    References
    ----------
    Ezaki et al. (2020) Nat Comm 3(1), 1-9.
    Fischer & Hertz (1991) "Spin glasses". Cambridge UP.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    if mu_width is not None:
        mu_actual = J.mean()
        x_range = np.arange(mu_actual - mu_width / 2,
                            mu_actual + mu_width / 2, 
                            mu_step)
    else:
        x_range = np.arange(*mu_range, mu_step)
    if std_width is not None:
        std_actual = J.std()
        y_range = np.arange(std_actual - std_width / 2,
                            std_actual + std_width / 2, 
                            std_step)
    else:
        y_range = np.arange(*std_range, std_step)
    diagram_shape = (len(x_range), len(y_range))
    
    m =      np.zeros(diagram_shape)
    q =      np.zeros(diagram_shape)
    chi_u =  np.zeros(diagram_shape)
    chi_sg = np.zeros(diagram_shape)
    C =      np.zeros(diagram_shape)

    if multiproc:
        if n_cpu is None:
            n_cpu = mp.cpu_count()
        pool = mp.Pool(n_cpu)
        all_mu = []
        all_stdev = []
        for mu in x_range:
            for stdev in y_range:
                all_mu.append(mu)
                all_stdev.append(stdev)
        results = pool.starmap(trans_and_metropolis, 
                               zip(repeat(h), repeat(J), repeat(beta), 
                                   all_mu, all_stdev,
                                   repeat(n_iter), repeat(rand_seed),
                                   repeat(save_every), repeat(save_path)))
        pool.close()
        for i in range(len(diagram_shape)):
            m[i // len(y_range), i % len(y_range)] = results[i][0]
            q[i // len(y_range), i % len(y_range)] = results[i][1]
            chi_u[i // len(y_range), i % len(y_range)] = results[i][2]
            chi_sg[i // len(y_range), i % len(y_range)] = results[i][3]
            C[i // len(y_range), i % len(y_range)] = results[i][4]
        
    else:
        for i, mu in enumerate(x_range):
            for j, stdev in enumerate(y_range):
                J_tilde = affine_trans(J, mu, stdev)
                (m[i, j], q[i, j], chi_u[i, j], chi_sg[i, j], C[i, j], 
                _, _) = metropolis(h, J_tilde, beta=beta, n_iter=n_iter, 
                                   rand_seed=rand_seed, rand_init=rand_init,
                                   save_every=save_every, save_path=save_path)
    
    return m, q, chi_u, chi_sg, C, x_range, y_range

def affine_trans(J, mu, stdev):
    """Linearly transform a square matrix to produce a new matrix with a 
    given mean and standard deviation.

    Parameters
    ----------
    J : 2d array (dtype='float')
        A square matrix.
    mu : float
        The new mean.
    stdev : float
        The new standard deviation.

    Returns
    -------
    J_tilde : 2d array (dtype='float')
        The transformed matrix with mean `mu` and standard deviation `stdev`.
        
    References
    ----------
    Ezaki et al. (2020) Nat Comm 3(1), 1-9.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    diag_mask = np.eye(J.shape[0], dtype=bool)    
    std_orig = np.std(J[~diag_mask])
    mu_orig = np.mean(J[~diag_mask])
    
    J_tilde = ( (J - ( np.ones(J.shape) - np.eye(J.shape[0]) ) * mu_orig)
                * stdev / std_orig
                + np.ones(J.shape) * mu )
        
    return J_tilde

def trans_and_metropolis(h, J, beta, mu, stdev, n_iter=1000, rand_seed=0, 
                         rand_init=False, save_every=None, save_path=None):
    """Linearly transform a square matrix to produce a new matrix with a 
    given mean and standard deviation, then sample data from the estimated 
    model using the Metropolis method and compute statistics.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.
    beta : float, optional
        The inverse temperature. 
    mu : float
        The new mean.
    stdev : float
        The new standard deviation.
    n_iter : int
        Number of iterations for Metropolis sampling.
    rand_seed : int
        Seed for random number generator.
    rand_init : bool
        If True, the initial spin configuration `sigma` will be randomly
        generated using rand_seed. If False, `sigma` will be all ones.
    
    Returns
    -------
    m : float
        Magnetization order parameter.
    q : float
        Spin-glass order parameter.
    chi_u : float
        Uniform susceptibility.
    chi_sg : float
        Spin-glass susceptiblity.
    C : float
        Specific heat.
    local_m : 2d array, dtype='float'
        Local mean.
    corr : 2d array, dtype='float'
        Correlation matrix.
    """
    J_tilde = affine_trans(J, mu, stdev)
    m, q, chi_u, chi_sg, C, local_m, corr = metropolis(h, 
                                                       J_tilde, 
                                                       beta,
                                                       n_iter=n_iter, 
                                                       rand_seed=rand_seed,
                                                       rand_init=rand_init,
                                                       save_every=save_every,
                                                       save_path=save_path)
    
    return m, q, chi_u, chi_sg, C, local_m, corr

def plot_sk(order_param, mu_values, sigma_values, J=None, title='SK diagram'):
    """Plot the SK diagram.
    
    Parameters
    ----------
    order_param : str
        The order parameter to plot, among {'m', 'q', 'chi_u', 'chi_sg', 'C'}
    mu_values : 1d array
        The mu (surrogate mean) values, on the x-axis.
    sigma_values : 1d array
        The sigma (surrogate standard deviation) values, on the y-axis
    J : 2d array of shape (N, N), optional
        The interaction matrix, for plotting the working point. 
    title : str, optional
        The plot title.
    """
    fig, ax = plt.subplots()
    ax.imshow(order_param, origin='lower', aspect='auto',
              extent=(mu_values[0], mu_values[-1], 
                      sigma_values[0], sigma_values[-1]))
    ax.set_xlabel('mean(J)')
    ax.set_ylabel('var(J)')
    ax.set_title(title)

    if J is not None:
        ax.plot(np.mean(J), np.std(J), marker='*', color='red', markersize=10)

    fig.show()

def ht_diagram(h, J, tail_cutoff,
               alpha_range=None, beta_range=None, mu_range=None, 
               sigma_range=None, n_iter=1000, metropolis_beta=1.0, 
               rand_seed=0, rand_init=True, save_every=None, save_path=None):
    """Produce "heavy-tail" phase diagram with 2 control parameters: 
    x is tail alpha or bulk mu;
    y is tail beta or bulk sigma.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.
    tail_cutoff : float
        Weight value which separates the bulk from the tail of the weight 
        distribution of J.
    alpha_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        tail offset.
    beta_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        tail slope.  
    mu_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        mean of the bulk weights.
    sigma_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        standard deviation of the bulk weights.
    n_iter : int, optional
        Number of iterations for Metropolis sampling. Default at 1000.
    metropolis_beta : float, optional
        The inverse temperature for Metropolis sampling. Default at unit 
        temperature 1.0.
    rand_seed : int, optional
        Seed for random number generator. Default at 0.
    rand_init : bool, optional
        If True, the initial spin configuration `sigma` for each simulation
        will be randomly generated using rand_seed. If False, `sigma` will be 
        all ones. Default at True.
    save_every : int, optional
        If not None, save the simulation every `save_every` iterations.
    save_path : str, optional
        Path to save the simulation.

    Returns
    -------
    m : 2d array
        Phase diagram in terms of the magnetization order parameter.
    q : 2d array
        Phase diagram in terms of the spin-glass order parameter..
    chi_u : 2d array
        Phase diagram in terms of the uniform susceptibility.
    chi_sg : 2d array
        Phase diagram in terms of the spin-glass susceptibility.
    C : 2d array
        Phase diagram in terms of the specific heat.
    x_range : 1d array
        Values for the mean.
    y_range : 1d array
        Values for the standard deviation.

    References
    ----------
    Ezaki et al. (2020) Nat Comm 3(1), 1-9.
    Fischer & Hertz (1991) "Spin glasses". Cambridge UP.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    # split tail from bulk
    tail_mask = J > tail_cutoff
    bulk = J[~tail_mask]
    bulk_mean = bulk.mean()
    bulk_std = bulk.std()

    if alpha_range is not None:
        alpha_values = np.arange(*alpha_range)
        x_range = alpha_values
        mu_values = np.ones_like(x_range) * bulk_mean # replace mean with mean
    elif mu_range is not None:
        mu_values = np.arange(*mu_range)
        x_range = mu_values
        alpha_values = np.zeros_like(x_range) # + 0
    if beta_range is not None:
        beta_values = np.arange(*beta_range)
        y_range = beta_values
        sigma_values = np.ones_like(y_range) * bulk_std # replace std with std
    elif sigma_range is not None:
        sigma_values = np.arange(*sigma_range)
        y_range = sigma_values
        beta_values = np.ones_like(y_range) # * 1
    diagram_shape = (len(alpha_values), len(beta_values))
    m =      np.zeros(diagram_shape)
    q =      np.zeros(diagram_shape)
    chi_u =  np.zeros(diagram_shape)
    chi_sg = np.zeros(diagram_shape)
    C =      np.zeros(diagram_shape)

    for i, (alpha, mu) in enumerate(zip(alpha_values, mu_values)):
        for j, (beta, sigma) in enumerate(zip(beta_values, sigma_values)):
            J_tilde = J.copy()
            # linear model
            J_tilde[tail_mask] = J_tilde[tail_mask] * beta + alpha
            # affine transformation
            J_tilde[~tail_mask] = (bulk - bulk_mean) * (sigma / bulk_std) + mu
            (m[i, j], q[i, j], chi_u[i, j], chi_sg[i, j], C[i, j], 
            _, _) = metropolis(h, J_tilde, beta=metropolis_beta, n_iter=n_iter, 
                                rand_seed=rand_seed, rand_init=rand_init,
                                save_every=save_every, save_path=save_path)
    
    return m, q, chi_u, chi_sg, C, x_range, y_range

def ising_energy(sigma, h, J):
    """Compute the energy of a spin configuration in the Ising model.

    Parameters
    ----------
    sigma : ndarray of shape (N,)
        Spin configuration vector where each element is +1 or -1.
    h : ndarray of shape (N,)
        External magnetic field vector.
    J : ndarray of shape (N, N)
        Interaction matrix between spins.

    Returns
    -------
    float
        The total energy of the given spin configuration.
    """
    return -0.5 * sigma @ J @ sigma - h @ sigma

def heat_capacity(data, h, J, mode='capacity'):
    """Compute the heat capacity (or specific heat) for a set of spin 
    configurations in the Ising model using energy variance.

    Parameters
    ----------
    data : ndarray of shape (N, M)
        Matrix of spin configurations, where N is the number of spins and
        M is the number of samples.
    h : ndarray of shape (N,)
        External magnetic field vector.
    J : ndarray of shape (N, N)
        Interaction matrix between spins.
    mode : {'capacity', 'specific'}, default='capacity'
        - 'capacity': returns the total heat capacity (variance of energies)
        - 'specific': returns the heat capacity per spin (specific heat)

    Returns
    -------
    float
        The heat capacity (or specific heat).
    """
    data_len = data.shape[1]
    E = np.zeros(data_len)
    for i in range(data_len):
        E[i] = ising_energy(data[:, i], h, J)
    heat = np.var(E)
    if mode == 'specific':
        heat = heat / data.shape[0]
    return np.var(E)

def energy_landscape_2d(sigma, length, flips, h, J):
    """Compute an arbitrary slice of the energy landscape for model {h, J}.

    Parameters
    ----------
    sigma : ndarray of shape (N,)
        Starting spin configuration vector where each element is +1 or -1.
    length : int
        Axis length of the square energy landscape.
    flips : ndarray of shape (2 * length,)
        The sequence of spin indices to flip. The first half is for axis 1,
        and the scond half for axis 2.
    h : ndarray of shape (N,)
        External magnetic field vector.
    J : ndarray of shape (N, N)
        Interaction matrix between spins.

    Returns
    -------
    el : ndarray of shape (length, length)
        The energy landscape spanned by the two spin-flip axes.
    """
    el = np.zeros((length, length))
    sigma = sigma.copy()

    for i in range(length):
        sigma[flips[i]] = -sigma[flips[i]]
        new_sigma = sigma.copy()
        for j in range(length):
            new_sigma[flips[length + j]] = -new_sigma[flips[length + j]]
            el[i, j] = ising_energy(new_sigma, h, J)

    return el