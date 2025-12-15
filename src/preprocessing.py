from math import isnan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .toptagging_patch import TopTaggingPatch

np.random.seed(1)

### -- MACROS --###

# invariant mass and energy in boost-centred system
m0 = 60
E0 = 600
# numerical precision
eps = 1e-6


### -- DATASET DOWNLOAD - ###

def download_toptagging_dataset(train_size, test_size):

    #select random subset from original datasets
    type_batch = 32
    total_train_size = 1200000
    train_mask_qcd = np.random.choice(np.arange(0, total_train_size - type_batch, type_batch*2), train_size, replace=False)
    train_mask_top = np.random.choice(np.arange(type_batch, total_train_size, type_batch*2), train_size, replace=False)
    train_mask = np.concatenate((train_mask_qcd, train_mask_top))
    train_mask.sort()

    total_test_size = 400000
    test_mask_qcd = np.random.choice(np.arange(0, total_test_size - type_batch, type_batch*2), test_size, replace=False)
    test_mask_top = np.random.choice(np.arange(type_batch, total_test_size, type_batch*2), test_size, replace=False)
    test_mask = np.concatenate((test_mask_qcd, test_mask_top))
    test_mask.sort()

    #label 0 for QCD, 1 for top
    particle_train, jet_train = TopTaggingPatch.getData(
        jet_type="all", data_dir="./.datasets/toptagging/", indices=train_mask, split="train", download=True
    )
    particle_test, jet_test = TopTaggingPatch.getData(
        jet_type="all", data_dir="./.datasets/toptagging/", indices=test_mask, split="test", download=True
    )

    return particle_train, jet_train, particle_test, jet_test


### -- AUXILIARY FUNCTIONS -- ###

def invariant_mass(p):
    mp = np.sqrt(p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2)
    if (isnan(mp)) | (mp == 0):
        return eps
    return mp

def Lorentz_boost(p, P):
    """
    Lorentz boost along P's spatial momentum and such that P's boosted energy is E0.

    args:
     - p: 4-momentum to transform;
     - P: 4-momentum along which the boost is performed.

    returns:
     - Lorentz @ p: p transformed according to the Lorentz boost matrix.
    """
    # p: 4-momentum to transform
    # P: 4-momentum along which the boost is performed
    gamma = (P[0]*E0 - np.sqrt(E0**2 - m0**2)*np.sqrt(P[1]**2 + P[2]**2 + P[3]**2))/m0**2
    beta_mod = np.sqrt(1 - 1/gamma**2)
    if P[0] > E0:
        beta = P[1:]*beta_mod/np.linalg.norm(P[1:])
    else:
        beta = -P[1:]*beta_mod/np.linalg.norm(P[1:])
    space_matx = np.identity(3) + (gamma - 1) * np.outer(beta, beta) / np.dot(beta, beta)
    Lorentz = np.zeros((4, 4))
    Lorentz[0,0] = gamma
    Lorentz[0,1:] = - gamma * beta
    Lorentz[1:,0] = - gamma * beta
    Lorentz[1:,1:] = space_matx
    return Lorentz @ p

def Gram_Schmidt_basis(p1, p2, p3):
    """
    Builds a Gram-Schmidt basis from p1, p2, p3.

    args:
     - p1, p2, p3: 3 biggest 4-momenta by spatial momentum to build the base from.

    returns:
     - e1, e2, e3: final Gram-Schmidt basis.
    """
    Pj = p1 + p2 + p3

    e1 = Pj[1:]/np.linalg.norm(Pj[1:])
    e2 = (p1[1:] - np.dot(p1[1:], e1) * e1)/ \
          np.linalg.norm(p1[1:] - np.dot(p1[1:], e1) * e1)
    e3 = (p2[1:] - np.dot(p2[1:], e1) * e1 - np.dot(p2[1:], e2) * e2)/ \
          np.linalg.norm(p2[1:] - np.dot(p2[1:], e1) * e1 - np.dot(p2[1:], e2) * e2)
    return e1, e2, e3

def jet_images(P_components):
    """
    Builds the jet histogram images by binning according to the projection
    on the Gram-Schmidt base vectors.

    args:
     - P_components: individual components of the boost. The first 3 are taken to build
       the Gram-Schmidt basis.

    returns:
     - energy_histogram: a 28x28 histogram of the spatial distribution of the boost energy
       partitioned among the components.
    """
    _, e2, e3 = Gram_Schmidt_basis(P_components[0], P_components[1], P_components[2])
    image_coordinates = lambda x : np.array([np.dot(x[1:]/np.linalg.norm(x[1:]), e2), np.dot(x[1:]/np.linalg.norm(x[1:]), e3)]) \
                                   if np.linalg.norm(x[1:]) > eps else \
                                   np.array([np.dot(x[1:]/eps, e2), np.dot(x[1:]/eps, e3)])
    image_coordinates = np.vectorize(image_coordinates, signature='(4)->(2)')
    P_components_XY = image_coordinates(P_components)
    bins = np.linspace(-1, 1, 29)
    # Clip the digitized results to be within the valid index range [0, 27]
    P_components_bins = np.clip(np.digitize(P_components_XY, bins), 0, 27)
    energy_histogram = np.zeros((28, 28))
    for i in range(len(P_components_bins)):
        energy_histogram[P_components_bins[i][0], P_components_bins[i][1]] += np.linalg.norm(P_components[i,1:])/E0
    return energy_histogram


### -- PREPROCESSING -- ###

def preprocessing(particle_train, particle_test, jet_train, jet_test, N_components):
    """
    Preprocessing pipeline. The following steps are performed:
     - sets are shuffled;
     - jets and components are rescaled so that the invariant mass for the jet is m0;
     - Lorentz boost is applied to the components to put them in a frame where
       spatial momentum orientation of the jet is the same, but total energy is E0;
     - jet images are created by projecting onto the Gram-Schmidt basis;
     - PCA is applied to reduce to N_components dimensions;
     - principal components are rescaled in a range [0, pi/2].

    args:
     - particle_train, particle_test: training and test set for the jets' components;
     - jet_train, jet_test: training and test set for the jets as a whole,
     - N_components: final number of components resulting from PCA.

    returns:
     - jet_train_pca, jet_y_train, jet_test_pca, jet_y_test: training and test PCA images, with
       respective labels.
    """
    shuffled_idx_train = np.random.permutation(len(particle_train))
    shuffled_idx_test = np.random.permutation(len(particle_test))

    particle_train = particle_train[shuffled_idx_train]
    particle_test = particle_test[shuffled_idx_test]
    jet_train = jet_train[shuffled_idx_train]
    jet_test = jet_test[shuffled_idx_test]

    jet_p_train = jet_train[:,1:]
    jet_p_test = jet_test[:,1:]
    jet_y_train = jet_train[:,0]
    jet_y_test = jet_test[:,0]

    #fill up empty jet 4-momenta with sum of all its components' 4-momenta
    for i in range(len(jet_p_train)):
        if np.all(jet_p_train[i] == 0):
            jet_p_train[i] = np.sum(particle_train[i], axis=0)

    for i in range(len(jet_p_test)):
        if np.all(jet_p_test[i] == 0):
            jet_p_test[i] = np.sum(particle_test[i], axis=0)

    # rescaling function
    # P: jet 4-momentum
    # p: 4-momentum of the particle/jet to rescale
    rescale = lambda P, p : p * m0 / invariant_mass(P)

    particle_scaled_train = np.zeros(particle_train.shape)
    particle_scaled_test = np.zeros(particle_test.shape)
    jet_p_scaled_train = np.zeros(jet_p_train.shape)
    jet_p_scaled_test = np.zeros(jet_p_test.shape)

    for i in range(len(particle_train)):
        for j in range(len(particle_train[i])):
            particle_scaled_train[i][j] = rescale(jet_p_train[i], particle_train[i][j])
        jet_p_scaled_train[i] = rescale(jet_p_train[i], jet_p_train[i])

    for i in range(len(particle_test)):
        for j in range(len(particle_test[i])):
            particle_scaled_test[i][j] = rescale(jet_p_test[i], particle_test[i][j])
        jet_p_scaled_test[i] = rescale(jet_p_test[i], jet_p_test[i])

    particle_boost_train = np.zeros(particle_train.shape)
    particle_boost_test = np.zeros(particle_test.shape)
    for i in range(len(particle_train)):
        for j in range(len(particle_train[i])):
            particle_boost_train[i][j] = Lorentz_boost(particle_scaled_train[i][j], jet_p_scaled_train[i])
    for i in range(len(particle_test)):
        for j in range(len(particle_test[i])):
            particle_boost_test[i][j] = Lorentz_boost(particle_scaled_test[i][j], jet_p_scaled_test[i])

    jet_train_images = np.zeros((len(particle_train), 28, 28))
    jet_test_images = np.zeros((len(particle_test), 28, 28))

    for i in range(len(particle_train)):
        jet_train_images[i] = jet_images(particle_boost_train[i])
    for i in range(len(particle_test)):
        jet_test_images[i] = jet_images(particle_boost_test[i])

    pca = PCA(n_components=N_components)
    jet_train_pca_unnorm = pca.fit_transform(jet_train_images.reshape(len(particle_train), -1))
    jet_test_pca_unnorm = pca.transform(jet_test_images.reshape(len(particle_test), -1))

    #normalisation
    normalizer = MinMaxScaler(feature_range=(0, np.pi/2))
    jet_train_pca = normalizer.fit_transform(jet_train_pca_unnorm)
    jet_test_pca = normalizer.transform(jet_test_pca_unnorm)

    return jet_train_pca, jet_y_train, jet_test_pca, jet_y_test
