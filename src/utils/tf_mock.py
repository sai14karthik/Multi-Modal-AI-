"""
Utility to suppress TensorFlow warnings and provide a lightweight stub so
`transformers` can be imported without the actual TensorFlow dependency.
"""

import os
import sys
import types
import importlib.util


def ensure_tensorflow_stub():
    """Stub TensorFlow to prevent import errors from optional deps."""
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

    sys.modules['tensorflow'] = tf_mock

