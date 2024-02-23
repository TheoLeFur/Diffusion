from trainer import train


def config():

    return dict(
        batch_size=32,
        n_channels=128,
        t_length=1000,
        n_res_blocks=2,
        n_epochs=1000,
        learning_rate=3e-4,
        beta1=1e-4,
        betaT=0.02
    )


if __name__ == "__main__":

    train(config())
