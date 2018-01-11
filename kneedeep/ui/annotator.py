from __future__ import print_function
from matplotlib.widgets import RectangleSelector
import numpy as np
from os.path import join, basename, dirname, splitext, isfile
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import close
import dicom
from scipy.misc import imread, imsave
from skimage import measure
import sys


class Annotator:
    def __init__(self, search_pattern, bbox=2, file_suffix='_mask', ext='.tif', qc=False):

        self.search_pattern = search_pattern
        self.file_suffix = file_suffix
        self.qc = qc
        self.bbox = bbox

        # Locate full paths to images and create output filenames
        self.input_imgs = [f for f in glob(search_pattern) if not f.endswith(file_suffix + ext)]  # ignore the masks
        self.output_paths = [join(dirname(x), splitext(basename(x))[0] + file_suffix + ext) for x in self.input_imgs]
        self.current_mask = None
        self.current_bbox = None
        self.bounding_boxes = []

    def annotate(self):

        # Display the figure window and first image to be annotated
        for index, (input_img, output_path) in enumerate(zip(self.input_imgs, self.output_paths)):

            if isfile(output_path):  # segmented file already exists

                if self.qc:
                    self.current_mask = imread(output_path)
                    no_blobs = np.max(measure.label(self.current_mask))
                    if no_blobs == 2:
                        continue
                    print("Previous segmentation invalid")
                else:
                    continue

            print(input_img)
            try:
                img_arr = dicom.read_file(input_img, force=True).pixel_array
            except Exception:
                img_arr = imread(input_img)

            self.current_mask = np.zeros(shape=img_arr.shape, dtype=np.uint8)

            self.fig, self.ax = plt.subplots(figsize=(18, 12))
            rs = RectangleSelector(self.ax, self.line_select_callback,
                                   drawtype='box', useblit=True,
                                   button=[1, 3],  # don't use middle button
                                   minspanx=5, minspany=5,
                                   spancoords='pixels',
                                   interactive=True)

            plt.connect('key_press_event', self.on_keypress)
            plt.imshow(img_arr, cmap='gray')
            plt.show()

            print("Saving mask")
            imsave(output_path, self.current_mask)
            self.bounding_boxes = []

    def progress(self):

        total_imgs = len(self.input_imgs)
        already_segmented = len([img for img in self.output_paths if isfile(img)])
        print("Total images available: {}".format(total_imgs))
        print("Images already annotated: {}".format(already_segmented))
        print("Images remaining: {}".format(total_imgs - already_segmented))

    def on_keypress(self, event):

        self.bounding_boxes.append(self.current_bbox)
        print(self.current_bbox)

        old_mask = np.copy(self.current_mask)

        if len(self.bounding_boxes) == self.bbox:

            for (x1, y1), (x2, y2) in self.bounding_boxes:
                self.current_mask[y1:y2, x1:x2] = 255

            no_blobs = np.max(measure.label(self.current_mask))

            if no_blobs != self.bbox:
                print("Invalid number of bounding boxes! Start over...")
                self.current_mask = old_mask
                self.bounding_boxes = []
            else:
                close()

            # if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            #     print(' RectangleSelector deactivated.')
            #     toggle_selector.RS.set_active(False)
            # if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            #     print(' RectangleSelector activated.')
            #     toggle_selector.RS.set_active(True)

    def line_select_callback(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.current_bbox = ((x1, y1), (x2, y2))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == '__main__':

    import sys
    ann = Annotator(sys.argv[1])
    ann.annotate()