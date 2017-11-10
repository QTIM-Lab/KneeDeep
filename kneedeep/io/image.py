from os import listdir
from os.path import isdir, join, basename
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from itertools import cycle


def save_figure(out_name, extensions=('.png', '.svg')):

    for ext in extensions:
        plt.savefig(out_name + ext, dpi=300)
        plt.savefig(out_name + ext, dpi=300)
    plt.close()


def save_overlay(background, mask, out_path, alpha=0.4, thresh=None):

    if thresh is not None:
        mask = (mask > thresh).astype(np.uint8)
        mask = np.ma.masked_where(mask == 0, mask)

    plt.figure()
    plt.imshow(background, 'gray', interpolation='none')
    plt.imshow(mask, 'reds', interpolation='none', alpha=alpha)
    plt.savefig(out_path)


def save_bb_overlay(background, bboxes, out_path):

    fig, ax = plt.subplots()
    ax.imshow(background, cmap='gray')

    colors = cycle(['c', 'm', 'y', 'k', 'b', 'g', 'r',  'w'])
    for min_row, min_col, max_row, max_col in bboxes:
        col = colors.next()
        rect = Rectangle((min_col, max_row), max_col - min_col, min_row - max_row, color=col, lw=1, fill=False)
        ax.add_patch(rect)

    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', dpi=300)


def progress_collage(pred_dir, out_path, background_images=None, thresh=0.9, alpha=0.5, no_epochs=75, extra_frames=10, no_samples=9, dpi=50):

    samples = sorted([join(pred_dir, x) for x in listdir(pred_dir) if isdir(join(pred_dir, x))],
                     key=lambda y: int(basename(y)))[:no_samples]
    no_epochs = min(len(listdir(samples[0])), no_epochs)
    no_rows_cols = int(np.floor(np.sqrt(len(samples))))
    samples = samples[:no_rows_cols*no_rows_cols]
    print '\n'.join(samples)

    if background_images is not None:
        raw_images = [mpimg.imread(join(background_images, '{}.bmp'.format(img))) for img in range(0, len(samples))]
    else:
        raw_images = None

    def update(i):

        plt.clf()

        i = min(i, no_epochs - 1)
        plt.gcf().text(0.02, 0.02, 'Epoch #{}'.format(i + 1), fontsize=24)

        for j in range(0, len(samples)):
            ax1 = plt.subplot(gs1[j])
            plt.axis('off')

            if raw_images is not None:
                ax1.imshow(raw_images[j], cmap='gray', interpolation=None)
                mask = (mpimg.imread(join(pred_dir, str(j), '{}.bmp'.format(i))) > thresh).astype(np.uint8)
                mask = np.ma.masked_where(mask == 0, mask)
                ax1.imshow(mask, cmap='Reds_r', alpha=alpha, interpolation=None)
            else:
                ax1.imshow(mpimg.imread(join(samples[j], '{}.bmp'.format(i))), cmap='gray', interpolation=None)

            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_xticks([])
            ax1.set_yticks([])

    fig = plt.figure(figsize=(16, 16))
    gs1 = gridspec.GridSpec(no_rows_cols, no_rows_cols, wspace=0.025, hspace=0.025)

    anim = FuncAnimation(fig, update, frames=no_epochs + extra_frames, interval=200)
    suffix = 'pred' if background_images is None else 'overlay'
    anim.save(join(out_path, 'animated_collage_{}.gif'.format(suffix)), dpi=dpi, writer='imagemagick')


