"""
Utility to suppress TensorFlow warnings and provide a lightweight stub so
`transformers` can be imported without the actual TensorFlow dependency.
"""

import os
import sys
import types
import importlib.util
import warnings


def ensure_tensorflow_stub():
    """Stub TensorFlow to prevent import errors from optional deps."""
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ.setdefault('WANDB_DISABLED', 'true')
    os.environ.setdefault('ACCELERATE_DISABLE_WEIGHTS_AND_BIASES', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # Suppress all FutureWarnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, message='.*bitsandbytes.*')
    warnings.filterwarnings('ignore', message='.*resume_download.*')
    warnings.filterwarnings('ignore', message='.*trust_remote_code.*')
    warnings.filterwarnings('ignore', message='.*text_config.*')
    warnings.filterwarnings('ignore', message='.*cadam32bit.*')
    warnings.filterwarnings('ignore', message='.*cadam32bit_grad_fp32.*')
    warnings.filterwarnings('ignore', message='.*text_config_dict.*')
    warnings.filterwarnings('ignore', message='.*CLIPTextConfig.*')
    warnings.filterwarnings('ignore', message='.*overriden.*')
    warnings.filterwarnings('ignore', message='.*id2label.*')
    warnings.filterwarnings('ignore', message='.*bos_token_id.*')
    warnings.filterwarnings('ignore', message='.*eos_token_id.*')
    
    # Suppress AttributeError warnings for cadam32bit
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    # Suppress torch distributed logging
    import logging
    logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').setLevel(logging.ERROR)

    if 'tensorflow' in sys.modules:
        return

    tf_mock = types.ModuleType('tensorflow')
    spec = importlib.util.spec_from_loader('tensorflow', loader=None)
    tf_mock.__spec__ = spec
    tf_mock.__version__ = '2.0.0'

    class MockTensor:
        pass

    class MockVariable:
        pass

    tf_mock.Tensor = MockTensor
    tf_mock.Variable = MockVariable
    tf_mock.image = types.ModuleType('tensorflow.image')
    tf_mock.nn = types.ModuleType('tensorflow.nn')
    tf_mock.io = types.ModuleType('tensorflow.io')
    tf_mock.io.gfile = types.ModuleType('tensorflow.io.gfile')

    def _noop(*_, **__):
        return None

    class _DummyGFile:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def read(self):
            return b""

        def write(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    tf_mock.io.gfile.GFile = _DummyGFile
    tf_mock.io.gfile.exists = lambda path: os.path.exists(path)
    tf_mock.io.gfile.listdir = lambda path: os.listdir(path) if os.path.isdir(path) else []
    tf_mock.io.gfile.makedirs = lambda path: os.makedirs(path, exist_ok=True)
    tf_mock.io.gfile.remove = lambda path: os.remove(path) if os.path.exists(path) else None
    tf_mock.io.gfile.join = staticmethod(os.path.join)
    tf_mock.io.gfile.IsDirectory = lambda path: os.path.isdir(path)

    sys.modules['tensorflow.io'] = tf_mock.io
    sys.modules['tensorflow.io.gfile'] = tf_mock.io.gfile

    if 'wandb' not in sys.modules:
        wandb_stub = types.ModuleType('wandb')
        wandb_stub.__spec__ = importlib.util.spec_from_loader('wandb', loader=None)

        def _wandb_noop(*args, **kwargs):
            class _DummyRun:
                def log(self, *a, **k):
                    return None

                def finish(self):
                    return None
            return _DummyRun()

        wandb_stub.init = _wandb_noop
        wandb_stub.login = _wandb_noop
        wandb_stub.finish = _wandb_noop

        wandb_sdk = types.ModuleType('wandb.sdk')
        wandb_sdk.__spec__ = importlib.util.spec_from_loader('wandb.sdk', loader=None)
        wandb_init = types.ModuleType('wandb.sdk.wandb_init')
        wandb_init.__spec__ = importlib.util.spec_from_loader('wandb.sdk.wandb_init', loader=None)
        wandb_init._attach = _wandb_noop
        wandb_init.init = _wandb_noop
        wandb_sdk.wandb_init = wandb_init
        wandb_sdk.wandb_login = types.ModuleType('wandb.sdk.wandb_login')
        wandb_sdk.wandb_login.__spec__ = importlib.util.spec_from_loader('wandb.sdk.wandb_login', loader=None)
        wandb_sdk.wandb_setup = types.ModuleType('wandb.sdk.wandb_setup')
        wandb_sdk.wandb_setup.__spec__ = importlib.util.spec_from_loader('wandb.sdk.wandb_setup', loader=None)
        wandb_sdk.wandb_settings = types.ModuleType('wandb.sdk.wandb_settings')
        wandb_sdk.wandb_settings.__spec__ = importlib.util.spec_from_loader('wandb.sdk.wandb_settings', loader=None)

        wandb_stub.sdk = wandb_sdk

        sys.modules['wandb'] = wandb_stub
        sys.modules['wandb.sdk'] = wandb_sdk
        sys.modules['wandb.sdk.wandb_init'] = wandb_init
        sys.modules['wandb.sdk.wandb_login'] = wandb_sdk.wandb_login
        sys.modules['wandb.sdk.wandb_setup'] = wandb_sdk.wandb_setup
        sys.modules['wandb.sdk.wandb_settings'] = wandb_sdk.wandb_settings

    sys.modules['tensorflow'] = tf_mock

