import seaborn as sns
from matplotlib import pyplot as plt
from qbstyles import mpl_style


def initialize():
    sns.set(rc={'figure.figsize': (22, 8.27)})
    mpl_style(dark=True)


def plot(df, column, smoothing=1, ax=None):
    rounds = len(df)
    clients = list(df[column])
    for client in clients:
        sns.lineplot(x=range(rounds), y=df[column][client].rolling(smoothing).mean(), color='r', alpha=0.2, ax=ax)
    sns.lineplot(x=range(rounds), y=df[column].quantile(0.25, axis=1).rolling(smoothing).mean(), color='orange',
                 alpha=0.5, ax=ax)
    sns.lineplot(x=range(rounds), y=df[column].median(axis=1).rolling(smoothing).mean(), color='orange', ax=ax)
    sns.lineplot(x=range(rounds), y=df[column].quantile(0.75, axis=1).rolling(smoothing).mean(), color='orange',
                 alpha=0.5, ax=ax)


def plot_processed(df, smoothing=1):
    rounds = len(df)
    clients = list(df)
    for client in clients:
        sns.lineplot(x=range(rounds), y=df[client].rolling(smoothing).mean(), color='r', alpha=0.2)
    sns.lineplot(x=range(rounds), y=df.quantile(0.25, axis=1).rolling(smoothing).mean(), color='orange',
                 alpha=0.5)
    sns.lineplot(x=range(rounds), y=df.median(axis=1).rolling(smoothing).mean(), color='orange')
    sns.lineplot(x=range(rounds), y=df.quantile(0.75, axis=1).rolling(smoothing).mean(), color='orange')


def get_params(df):
    return set([param[0][3:] for param in list(df) if '.' in param[0] and 'l2' in param[0]])


def plot_l2_norms(df, smoothing=1):
    f, axes = plt.subplots(4, 2, sharex=True, figsize=(30,20))
    for param, (i,j) in zip(get_params(df),zip(list(range(4))*2, [0]*4 + [1] *4)):
        plot(df, "l2_"+param, smoothing=smoothing, ax=axes[i,j])
        axes[i, j].set_title(param)
    for ax in axes.flat:
        ax.set(xlabel='Round', ylabel='L2 Norm')
    plt.subplots_adjust(hspace=0.5)


def plot_cosine_similarity(df, smoothing=1):
    f, axes = plt.subplots(4, 2, sharex=True, figsize=(30,20))
    for param, (i,j) in zip(get_params(df),zip(list(range(4))*2, [0]*4 + [1] *4)):
        plot(df, "avg_cosine_"+param, smoothing=smoothing, ax=axes[i,j])
        axes[i, j].set_title(param)
    for ax in axes.flat:
        ax.set(xlabel='Round', ylabel='Avg Cosine Similarity')
    plt.subplots_adjust(hspace=0.5)
