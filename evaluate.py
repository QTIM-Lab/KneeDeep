from os import chdir
from os.path import join
import yaml
import pandas as pd
from kneedeep.models import KneeLocalizer
from kneedeep.io.image import progress_collage, save_figure
from kneedeep.io.paths import makedir_if_not_exists


def evaluate(config, model, data):

    eval_dir = join(config['output_dir'], 'evaluate')
    makedir_if_not_exists(eval_dir)

    # Summarize the training
    history = pd.DataFrame.from_csv('history.csv')
    chdir(eval_dir)

    history.plot(y=['loss', 'val_loss'], kind='line')
    save_figure('training_loss')

    history.plot(y=['dice_coef', 'val_dice_coef'], kind='line')
    save_figure('training_dice')

    # Evaluate performance
    joint_localizer = KneeLocalizer(config, pretrained_model=model)
    summary, dice_scores = joint_localizer.evaluate(data, eval_dir)
    summary.to_csv('performance_metrics.csv')
    dice_scores.to_csv('dice_scores.csv')

    # Make collage
    make_collage(config)


def make_collage(config):

    # Generate progress collage animation
    progress_dir = join(config['output_dir'], 'progress')
    pred_dir = join(progress_dir, 'prediction')
    raw_dir = join(progress_dir, 'raw')

    progress_collage(pred_dir, progress_dir, background_images=None, no_samples=16)
    progress_collage(pred_dir, progress_dir, background_images=raw_dir, no_samples=16)


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    with open(parser.parse_args().config, 'rb') as f:
        config = yaml.load(f)

    evaluate(config, 'final_model.h5', config['save_name'] + '.h5')


