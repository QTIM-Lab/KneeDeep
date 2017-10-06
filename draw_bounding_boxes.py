import sys
from qtim_OA.ui.annotator import Annotator

if __name__ == '__main__':

    ann = Annotator(sys.argv[1], qc=True)
    ann.progress()
