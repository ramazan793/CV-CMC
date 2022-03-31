import os
import pickle
import re

import numpy as np

import interface


def load_test_data(test_name, test_path):
    return pickle.load(open(f'{test_path}/{test_name}.pickle', 'rb'))


def unpack_ndarray_as_dtype(constructor, args, state, dtype):
    arr = constructor(*args)
    arr.__setstate__(state)
    return arr.astype(dtype)


# region Generic tests
def check_interface(impl, interface_base):
    with SubTest(
        impl.__name__
    ):
        this_impl = f"The {impl.__name__} class "
        this_interface = f" {interface_base.__name__} abstract base class."
        if not issubclass(impl, interface_base):
            raise ImplementationError(
                this_impl + "should inherit from the" + this_interface
            )
        for method_name in interface_base.__abstractmethods__:
            method = getattr(impl, method_name, None)
            if not callable(method):
                raise ImplementationError(
                    this_impl +
                    f"doesn't have the {method_name} method, required by" +
                    this_interface
                )
            if getattr(method, '__isabstractmethod__', False):
                raise ImplementationError(
                    this_impl +
                    f"doesn't implement the {method_name} method, required by" +
                    this_interface
                )


def init_layer(layer_impl, test_data):
    # Build layer
    layer = layer_impl(**test_data['kwargs'])
    layer.build(MockOptimizer())
    extra_info = {}

    # Set parameters of layer
    for k, v in test_data.items():
        if k.startswith('parameter_'):
            parameter_name = k[len('parameter_'):]
            param = getattr(layer, parameter_name, None)

            with SubTest(
                f'layer.{parameter_name}',
                extra_info=extra_info,
            ):
                # Parameter must be already initialized in the build method of
                # the Layer and must have correct shape/dtype
                assert_ndarray_compatible(
                    actual=param,
                    correct=v,
                )

            # Prepare the initial value of the parameter for test
            param[:] = v
            extra_info[f'layer.{parameter_name}'] = v

    # Test forward pass
    extra_info['inputs'] = test_data['inputs']

    # Test in evaluation mode
    if 'is_training' in test_data:
        extra_info[f'layer.is_training'] = test_data['is_training']
        layer.is_training = test_data['is_training']

    # Test non-deterministic layers
    if 'seed' in test_data:
        extra_info['Numpy PRNG seed'] = test_data['seed']
        np.random.seed(test_data['seed'])

    return layer, extra_info


def forward_layer(layer_impl, test_data):
    layer, extra_info = init_layer(layer_impl, test_data)

    # Test forward pass
    with SubTest(
        f'layer.forward(inputs)',
        extra_info=extra_info
    ):
        assert_ndarray_equal(
            actual=layer.forward(test_data['inputs']),
            correct=test_data['outputs']
        )

    # Test, that running parameters were updated correctly
    for k, parameter_value in test_data.items():
        if k.startswith('after_'):
            parameter_name = k[len('after_'):]
            with SubTest(
                f"layer.{parameter_name}",
                extra_info=extra_info
            ):
                if parameter_name in {'running_mean', 'running_var'}:
                    continue
                assert_ndarray_equal(
                    actual=getattr(layer, parameter_name, None),
                    correct=parameter_value
                )


def backward_layer(layer_impl, test_data):
    layer, extra_info = init_layer(layer_impl, test_data)
    with SubTest(
        f"layer.forward(inputs)",
        extra_info=extra_info
    ):
        layer.forward(test_data['inputs'])

    # Test backward pass
    extra_info["grad_outputs"] = test_data['grad_outputs']
    with SubTest(
        f"layer.backward(grad_outputs)",
        extra_info=extra_info
    ):
        assert_ndarray_equal(
            actual=layer.backward(test_data['grad_outputs']),
            correct=test_data['grad_inputs']
        )

    # Test, that parameter gradients were calculated correctly
    for k, grad_value in test_data.items():
        if k.startswith('param_grad_'):
            grad_name = k[len('param_grad_'):] + '_grad'
            with SubTest(
                f"layer.{grad_name}",
                extra_info=extra_info
            ):
                assert_ndarray_equal(
                    actual=getattr(layer, grad_name, None),
                    correct=grad_value
                )


def loss(loss_impl, test_data, method):
    loss_inst = loss_impl()
    with SubTest(
        f'loss.{method}(y_gt, y_pred)',
        extra_info={
            'y_gt': test_data['y_gt'],
            'y_pred': test_data['y_pred'],
        },
    ):
        assert_ndarray_equal(
            actual=getattr(loss_inst, method)(
                test_data['y_gt'], test_data['y_pred']
            ),
            correct=test_data[method]
        )


def function(function_impl, test_data):
    with SubTest(
        f"{function_impl.__name__}({', '.join(test_data['kwargs'])})",
        extra_info=test_data['kwargs'],
    ):
        outputs = function_impl(**test_data['kwargs'])
        assert_ndarray_equal(
            actual=outputs,
            correct=test_data['outputs']
        )


def simulate_optimizer(optimizer_impl, test_data):
    optimizer = optimizer_impl(**test_data['kwargs'])
    updaters = []
    for sh in test_data['parameter_shapes']:
        with SubTest(
            "optimizer.get_parameter_updater(shape)",
            extra_info={"shape": sh},
        ):
            updaters.append(optimizer.get_parameter_updater(sh))

    for step in test_data['steps']:
        for param_data, updater in zip(step, updaters):
            with SubTest(
                "updater(parameter, parameter_grad)",
                extra_info={
                    "parameter": param_data['value'],
                    "parameter_grad": param_data['grad']
                },
            ):
                assert_ndarray_equal(
                    actual=updater(param_data['value'], param_data['grad']),
                    correct=param_data['new_value']
                )


# endregion


# region Generic asserts & Pretty printing
try:
    term_width = os.get_terminal_size().columns
except OSError:
    term_width = 80


def array_repr_oneline(*args, **kwargs):
    value = original_array_repr(*args, **kwargs)
    return re.sub(r'\s+', ' ', value, re.M)


# noinspection PyUnresolvedReferences
original_array_repr = np.core.array_repr
np.core.array_repr = array_repr_oneline
np.set_string_function(array_repr_oneline)


class SubTest(object):
    def __init__(self, target, extra_info=None):
        self.target = target
        self.extra_info = extra_info

    def get_extra_info(self):
        if self.extra_info:
            extra_info = "\n".join(
                f"    {k}={repr(v)}" for k, v in self.extra_info.items()
            )
            return (
                "\n >>> Test Initialization <<<\n"
                "Before running the test, the following "
                "initial values were set:\n" +
                extra_info +
                "\n"
            )
        else:
            return ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        __tracebackhide__ = True

        if exc_type is None:
            return

        if issubclass(
            exc_type,
            (WrongTypeError, WrongDTypeError, WrongShapeError, WrongValueError)
        ):
            reason = exc_val.reason
            details = getattr(exc_val, 'details', "")
        else:
            reason = "Something went wrong during the {target} test"
            details = repr(exc_val)
            if hasattr(exc_val, 'msg'):
                del exc_val.msg

        extra_info = self.get_extra_info()
        error = (
            "\n !!! " + reason + " !!!\n" +
            extra_info +
            "\n >>> Details <<<\n" +
            details
        )
        error = error.format(target=self.target)

        exc_val.args = (error,)
        return False


class CustomAssertionError(AssertionError):
    reason = "Something went wrong during the {target} test"
    target_prefix = ""
    target_postfix = ""

    def __init__(self, actual, correct):
        full_target = f"{self.target_prefix}{{target}}{self.target_postfix}"
        self.details = (
            f"\nYour implementation produced\n    {full_target}=" +
            str(actual) +
            "\n"
            f"\nThe correct answer is\n    {full_target}=" +
            str(correct) +
            "\n"
        )


class ImplementationError(CustomAssertionError):
    reason = "The implementation of {target} is wrong"

    def __init__(self, details):
        self.details = details


class WrongTypeError(CustomAssertionError):
    reason = "The type of {target} is wrong"
    target_prefix = "type("
    target_postfix = ")"


class WrongDTypeError(CustomAssertionError):
    reason = "The data type (dtype) of {target} is wrong"
    target_postfix = ".dtype"


class WrongShapeError(CustomAssertionError):
    reason = "The shape of {target} is wrong"
    target_postfix = ".shape"


class WrongValueError(CustomAssertionError):
    reason = "The value of {target} is wrong"

    def __init__(self, details):
        details = "\n" + details
        details = details.replace(
            "x: ", "\nYour implementation produced\n    {target}="
        )
        details = details.replace(
            "y: ", "\nThe correct answer is\n    {target}="
        )
        details = details.lstrip()
        self.details = details


def assert_value_is_ndarray(value):
    __tracebackhide__ = True
    if not isinstance(value, (np.ndarray, np.generic)):
        raise WrongTypeError(type(value).__name__, np.ndarray.__name__)


def assert_dtypes_compatible(actual_dtype, correct_dtype):
    __tracebackhide__ = True
    if not (
        np.can_cast(actual_dtype, correct_dtype, casting='same_kind') and
        np.can_cast(correct_dtype, actual_dtype, casting='same_kind')
    ):
        raise WrongDTypeError(actual_dtype, correct_dtype)


def assert_shapes_match(actual_shape, correct_shape):
    __tracebackhide__ = True
    if not (
        len(actual_shape) == len(correct_shape) and
        actual_shape == correct_shape
    ):
        raise WrongShapeError(actual_shape, correct_shape)


def assert_ndarray_compatible(actual, correct):
    __tracebackhide__ = True
    assert_value_is_ndarray(actual)
    assert_dtypes_compatible(actual.dtype, correct.dtype)
    assert_shapes_match(actual.shape, correct.shape)


def assert_ndarray_equal(*, actual, correct):
    __tracebackhide__ = True
    assert_ndarray_compatible(actual, correct)
    exc = None
    try:
        np.testing.assert_allclose(actual, correct, rtol=1e-5, verbose=True)
    except AssertionError as e:
        exc = e
    if exc is not None:
        raise WrongValueError(*exc.args)


# endregion


# region Mocks
class MockOptimizer(interface.Optimizer):
    """Fake optimizer, that doesn't update the parameters"""

    def get_parameter_updater(self, shape):
        def update(parameter, parameter_grad):
            return parameter

        return update

# endregion
