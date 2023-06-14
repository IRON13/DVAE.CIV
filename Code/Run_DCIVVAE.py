import argparse
import logging
import pandas as pd
import torch
import pyro
from DCIVVAE_gpu import DCIVVAE
from datasets import load_dataset
import numpy as np
from econml.iv.nnet import DeepIV
import keras

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


def main(args, reptition, path, sample_size):
    pyro.enable_validation(__debug__)
    # if args.cuda:
    torch.set_default_tensor_type('torch.FloatTensor')

    # Generate synthetic data.
    pyro.set_rng_seed(args.seed)
    train, test, contfeats, binfeats = load_dataset(path = path, reps = reptition, cuda = False, sample= sample_size)
    (x_train, t_train, y_train), true_ite_train = train
    (x_test, t_test, y_test), true_ite_test = test
    
    ym, ys = y_train.mean(), y_train.std()
    y_train = (y_train - ym) / ys

    # Train.
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    model = DCIVVAE(feature_dim=args.feature_dim, continuous_dim= contfeats, binary_dim = binfeats,
                  latent_dim=args.latent_dim, latent_dim_t = args.latent_dim_t, latent_dim_y = args.latent_dim_y,
                  hidden_dim=args.hidden_dim,
                  num_layers=args.num_layers,
                  num_samples=10)                                                                                                                                                                                                                                                                                                                                                           
    model.fit(x_train, t_train, y_train,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              learning_rate_decay=args.learning_rate_decay, weight_decay=args.weight_decay)

    zt_test = model.guide.zt(x_test)
    zc_test = model.guide.zc(x_test)
    zy_test = model.guide.zy(x_test)

    zt_train = model.guide.zt(x_train)
    zc_train = model.guide.zc(x_train)
    zy_train = model.guide.zy(x_train)

    zt_test = zt_test.cpu().detach().numpy().astype(np.float16)
    zc_test = zc_test.cpu().detach().numpy().astype(np.float16)
    zy_test = zy_test.cpu().detach().numpy().astype(np.float16)
    zt_train = zt_train.cpu().detach().numpy().astype(np.float16)
    zc_train = zc_train.cpu().detach().numpy().astype(np.float16)
    zy_train = zy_train.cpu().detach().numpy().astype(np.float16)

    condition_x_test = np.hstack((zc_test, zy_test))
    condition_x_train = np.hstack((zc_train, zy_train))

    treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(6,)),
                                        keras.layers.Dense(64, activation='relu'),
                                        keras.layers.Dense(32, activation='relu')])

    response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(6,)),
                                       keras.layers.Dense(64, activation='relu'),
                                       keras.layers.Dense(32, activation='relu'),
                                       keras.layers.Dense(1)])

    keras_fit_options = { "epochs": 300,
                          "validation_split": 0.1,
                          "callbacks": [keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]}

    deepIvEst = DeepIV(n_components = 10, # number of gaussians in our mixture density network
                       m = lambda z, x : treatment_model(keras.layers.concatenate([z,x])), # treatment model
                       h = lambda t, x : response_model(keras.layers.concatenate([t,x])),  # response model
                       n_samples = 1, # number of samples to use to estimate the response
                       use_upper_bound_loss = False, # whether to use an approximation to the true loss
                       n_gradient_samples = 1, # number of samples to use in second estimate of the response (to make loss estimate unbiased)
                       optimizer='adam', # Keras optimizer to use for training - see https://keras.io/optimizers/
                       first_stage_options = keras_fit_options, # options for training treatment model
                       second_stage_options = keras_fit_options) # options for training response model

    deepIvEst.fit(Y=y_train,T=t_train,X=condition_x_train,Z=zt_train)



    effect_train = (deepIvEst.predict(T=torch.ones(int(sample_size*0.7)),X=condition_x_train) - deepIvEst.predict(T=torch.zeros(int(sample_size*0.7)),X=condition_x_train))
    pehe_train = np.sqrt(
        np.mean((true_ite_train.squeeze() - effect_train) * (true_ite_train.squeeze() - effect_train)))
    ATE_train = abs(np.mean(true_ite_train.squeeze()) - np.mean(effect_train))

    effect_test = (
        deepIvEst.predict(T=torch.ones(int(sample_size*0.3)), X=condition_x_test) - deepIvEst.predict(T=torch.zeros(int(sample_size*0.3)), X=condition_x_test))
    pehe_test = np.sqrt(
        np.mean((true_ite_test.squeeze() - effect_test) * (true_ite_test.squeeze() - effect_test)))
    ATE_test = abs(np.mean(true_ite_test.squeeze()) - np.mean(effect_test))

    return pehe_train, pehe_test, ATE_train, ATE_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DCIVVAE")
    parser.add_argument("--feature-dim", default=6, type=int)
    parser.add_argument("--latent-dim", default=3, type=int)
    parser.add_argument("--latent-dim-t", default=1, type=int)
    parser.add_argument("--latent-dim-y", default=2, type=int)
    parser.add_argument("--hidden-dim", default=100, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("-n", "--num-epochs", default=128, type=int)
    parser.add_argument("-b", "--batch-size", default=512, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=1234567890, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    N_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]
    for sample_size in N_list:
        path = './Data/Syn_' + str(sample_size)
        res_pehe_train = []
        res_pehe_test = []
        res_ATE_train = []
        res_ATE_test = []

        for i in range(30):
            print("Dataset {:d}".format(i + 1))
            pehe_train, pehe_test, ATE_train, ATE_test = main(args, i + 1, path,sample_size)
            res_pehe_train.append(pehe_train)
            res_pehe_test.append(pehe_test)
            res_ATE_train.append(ATE_train)
            res_ATE_test.append(ATE_test)

        train_1 = pd.DataFrame(res_pehe_train)
        test_1 = pd.DataFrame(res_pehe_test)
        train_2 = pd.DataFrame(res_ATE_train)
        test_2 = pd.DataFrame(res_ATE_test)

        result = pd.concat([train_1, test_1, train_2, test_2], axis=1)
        result.to_csv(r'./Res/DCIVVAE_deepiv_' + str(sample_size) + '.csv', sep=',', float_format='%.5f')


