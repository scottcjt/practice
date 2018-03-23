import os
import pathlib as pl

import click

import lang8
import model


LOG_FOLDER = 'logs'
CKPT_FOLDER = 'ckpts'
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('run_name', nargs=1)
@click.option('--corpus', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--epochs', default=2)
@click.option('--batch_size', default=128)
@click.option('--embed_size', default=128)
@click.option('--hidden_size', default=128)
def main(run_name, corpus, epochs, batch_size, embed_size, hidden_size):
    workspace = pl.Path(os.path.realpath(__file__)).parent
    logdir = workspace / LOG_FOLDER / run_name
    ckptdir = workspace / CKPT_FOLDER / run_name

    logdir.mkdir(mode=0o755, parents=True, exist_ok=False)
    ckptdir.mkdir(mode=0o755, parents=True, exist_ok=False)

    print('Summary folder: {}'.format(logdir))
    print('Checkpoint folder: {}'.format(ckptdir))
    print('Use corpus {}, train for {} epochs'.format(corpus, epochs))

    print('Loading corpus...')
    feeder = lang8.Lang8Data(corpus, corpus + '_vocabs')
    print('Building model...')
    m = model.GrammarCorrectionModel(
        feeder.start_symbol, feeder.end_symbol, feeder.pad_symbol,
        batch_size, embed_size, hidden_size,
        str(logdir), str(ckptdir)
    )

    print('Start...')
    m.run(feeder, epochs)


if __name__ == '__main__':
    exit(main())
