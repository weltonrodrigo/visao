import acapture
from argparse import ArgumentParser
import sys
import logging
import yaml
from viewer import PyGLViewer
import state

ll = None

class Calibrator():

    _config = {
        'cbrow': 7,
        'cbcol': 10,
        'samples': 30,
        'delay': .5,
        'load': False,
        'write': False,
        'file': False,
        'alpha': 0
    }

    @property
    def config(self):
        return self._config

    def __init__(self, **kwargs):
        self.state = None
        self._cap = None
        self._viewer = PyGLViewer()

        for key, value in kwargs.items():
            self._config[key] = value

    def transition_to(self, state, context={}):
        ll.info(f'Transitioning to {state} state')
        self._state = state
        state.context_data = context
        state.calibrator = self

    def read(self):
        """Read a frame from the capture device"""
        return self._cap.read()

    def set_image(self, image):
        """Set a frame into the displaying device"""
        return self._viewer.set_image(image)

    def set_loop(self, loop):
        self._viewer.set_loop(loop)

    def set_keyboard_listener(self, func):
        self._viewer.add_keyboard_listener(func)

    def write_params(self, params: {}):
        if not self.config['write']:
            return
        else:
            file = self.config['write']
            ll.info('Will write camera params to file %s', file)
            with open(file, mode='w') as f:
                yaml.dump(params, stream=f, Dumper=yaml.Dumper)

    def read_params(self) -> {}:
        file = self.config['read']
        ll.info('Will read camera params from file %s' % file)
        with open(file, mode='r') as f:
            p = yaml.load(f, Loader=yaml.Loader)
            ll.info('Read camera parameters from file: %s' % p)
            return p

    def start(self, device):
        self._cap = acapture.open(device)

        if self.config['read']:
            camera_params = self.read_params()
            context = {
                'camera_params': camera_params
            }
            self.transition_to(state.StateCalibrated(), context)
        else:
            self.transition_to(state.StateInitial())

        self._viewer.start()


def setup_logging(debug: bool):
    global ll
    level = logging.DEBUG if debug else logging.INFO

    ll = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(filename)s:%(lineno)d] %(message)s"))
    ll.addHandler(handler)
    ll.setLevel(level)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-w', '--write', help='write calibration data')
    p.add_argument('-r', '--read', help='read calibration data')
    p.add_argument('--delay', type=float, default=0.5,
                   help='delay between samples when calibrating')
    p.add_argument('--samples', type=float, default=30,
                   help='how many samples to take when calibrating')
    p.add_argument('--alpha', type=int, default=0,
                   help='use all pixels in new matrix')
    p.add_argument('--images', nargs='+', help='calibrate from images')
    p.add_argument('-d', action='store_true', help='debug logs')

    args = p.parse_args()

    setup_logging(args.d)

    calib = Calibrator(**vars(args))
    calib.start(0)
