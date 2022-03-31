import abc
# noinspection PyUnresolvedReferences
import os
import time
import warnings

import numpy as np

try:
    import tqdm
except ImportError:
    tqdm = None

np.seterr(all='raise', under='ignore')
eps = np.finfo(np.float64).eps

_model_fstring = "{name:>20} | total parameters: {params:>20,}"
_layer_fstring = (
    "{name:>20} | input: {inp:<20} output: {out:<20} | params: {params:>16,}"
)


# region Abstract base classes
class Layer(abc.ABC):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.output_shape = None

        self.forward_inputs = None
        self.forward_outputs = None
        self.is_training = True

        self._parameter_updaters = {}
        self._optimizer = None
        self._is_built = False

    def __str__(self):
        return _layer_fstring.format(
            name=str(self.__class__.__name__),
            inp=str(self.input_shape),
            out=str(self.output_shape),
            params=self.num_parameters(),
        )

    def build(self, optimizer, prev_layer=None):
        self._optimizer = optimizer
        if prev_layer is not None:
            self.input_shape = prev_layer.output_shape
        elif self.input_shape is None:
            raise ValueError(
                'Unable to infer the input shape for '
                f'layer {self.__class__.__name__}. '
                'If this is the first layer in the model, '
                'please specify the "input_shape" parameter.'
            )
        self.output_shape = (
            self.input_shape if self.output_shape is None
            else self.output_shape
        )
        self._is_built = True

    def add_parameter(self, name, shape, initializer):
        if not self._is_built:
            raise RuntimeError(
                "add_parameter must be called after build "
                "(or after super().build inside custom build method)"
            )
        self._parameter_updaters[name] = \
            self._optimizer.get_parameter_updater(shape)
        param = initializer(shape)
        grad = np.zeros(shape)
        if param.shape != grad.shape:
            raise RuntimeError(
                "Something went wrong in add_parameter: "
                "the initializer returned a tensor of incorrect shape."
            )
        if len(shape) == 4:
            # Store the parameter tensors for convolutional layers backwards in
            # memory to avoid a copy in _numpy_to_torch during forward pass in
            # convolve_pytorch. This is just a minor performance optimization
            # and doesn't change the actual computation.
            param = param[:, :, ::-1, ::-1]
        return param, grad

    def num_parameters(self):
        return sum(
            getattr(self, name).size for name in self._parameter_updaters.keys()
        )

    def update_parameters(self):
        for name, updater in self._parameter_updaters.items():
            for k in (name, name + '_grad'):
                if not hasattr(self, k):
                    raise AttributeError(
                        f"Parameter {name} was registered for "
                        f"{self.__class__.__name__}, but attribute self.{k} "
                        "doesn't exits."
                    )
            parameter = getattr(self, name)
            parameter_grad = getattr(self, name + '_grad')

            if np.isnan(parameter).any():
                raise ValueError(
                    f"During parameter update, the parameter {name} of "
                    f"Layer {self.__class__.__name__} has NaN values."
                )

            if np.isnan(parameter_grad).any():
                raise ValueError(
                    "During parameter update, the gradient of the parameter "
                    f"{name} of Layer {self.__class__.__name__} has NaN values."
                )

            # Updater shouldn't directly modify parameter and gradient arrays
            parameter.flags.writeable = False
            parameter_grad.flags.writeable = False

            new_parameter = updater(parameter, parameter_grad)

            parameter.flags.writeable = True
            parameter_grad.flags.writeable = True

            parameter[:] = new_parameter

            if np.isnan(parameter).any():
                raise ValueError(
                    f"After parameter update, the parameter {name} of "
                    f"Layer {self.__class__.__name__} has NaN values."
                )

    def forward(self, inputs):
        inputs = np.ascontiguousarray(inputs)
        inputs.flags.writeable = False
        self.forward_inputs = inputs

        outputs = self.forward_impl(inputs)

        outputs = np.ascontiguousarray(outputs)
        outputs.flags.writeable = False
        self.forward_outputs = outputs

        return outputs

    def backward(self, grad_outputs):
        grad_outputs = np.ascontiguousarray(grad_outputs)
        grad_outputs.flags.writeable = False

        grad_inputs = self.backward_impl(grad_outputs)

        grad_inputs = np.ascontiguousarray(grad_inputs)
        grad_inputs.flags.writeable = False

        self.forward_inputs = None
        self.forward_outputs = None

        return grad_inputs

    @abc.abstractmethod
    def forward_impl(self, inputs):
        pass

    @abc.abstractmethod
    def backward_impl(self, grad_outputs):
        pass


class Loss(abc.ABC):
    def value(self, y_gt, y_pred):
        y_gt.flags.writeable = False
        y_pred.flags.writeable = False

        value = self.value_impl(y_gt, y_pred)

        value.flags.writeable = False

        return value

    def gradient(self, y_gt, y_pred):
        y_gt.flags.writeable = False
        y_pred.flags.writeable = False

        grad = self.gradient_impl(y_gt, y_pred)

        grad.flags.writeable = False

        return grad

    @abc.abstractmethod
    def value_impl(self, y_gt, y_pred):
        pass

    @abc.abstractmethod
    def gradient_impl(self, y_gt, y_pred):
        pass


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def get_parameter_updater(self, shape):
        pass


# endregion

# region Boilerplate
def he_initializer(input_dim):
    def _he_initializer(shape):
        return np.random.randn(*shape) * np.sqrt(2.0 / input_dim)

    return _he_initializer


def range_fn(*args, **kwargs):
    if tqdm is None:
        return range(*args), print, lambda: None, lambda desc: None
    else:
        progress = tqdm.tqdm(**kwargs)

        def set_desc(desc):
            progress.desc = desc

        def write(*a, **kw):
            progress.write(*a, **kw)
            if progress.n == progress.total:
                progress.close()

        return range(*args), write, progress.update, set_desc


class Model(object):
    def __init__(self, loss, optimizer):
        if not isinstance(loss, Loss):
            raise RuntimeError(
                "Model loss should be an instance of Loss class. "
                f"Instead got: {loss} "
                f"of type {loss.__class__.__name__}."
            )
        if not isinstance(optimizer, Optimizer):
            raise RuntimeError(
                "Model optimizer should be an instance of Optimizer class. "
                f"Instead got: {optimizer} "
                f"of type {optimizer.__class__.__name__}."
            )
        self._layers = []
        self._loss = loss
        self._optimizer = optimizer
        self._last_y_pred = None

        self.loss_train_history = []
        self.loss_valid_history = []
        self.accuracy_train_history = []
        self.accuracy_valid_history = []

    def add(self, layer):
        if not self._layers:
            layer.build(self._optimizer)
        else:
            layer.build(self._optimizer, prev_layer=self._layers[-1])
        self._layers.append(layer)

    def forward(self, x_gt, training=False, verbose=False):
        output = x_gt

        output_meaning = "the network input shape"
        timings = []

        for idx, layer in enumerate(self._layers):
            name = layer.__class__.__name__
            if layer.input_shape != output.shape[1:]:
                raise ValueError(
                    "In forward pass, the input shape of "
                    f"Layer {name} "
                    f"doesn't match {output_meaning}:\n\t"
                    f"layer_expected_input.shape: {layer.input_shape}, "
                    f"layer_actual_input.shape: {output.shape[1:]}"
                )
            output_meaning = "the output shape of previous layer"
            layer.is_training = training

            forward_start = time.time()
            output = layer.forward(output)
            timings.append((name, time.time() - forward_start))

            if np.isnan(output).any():
                raise ValueError(
                    "In forward pass, the output of "
                    f"Layer {name} has NaN values."
                )

            if layer.output_shape != output.shape[1:]:
                raise ValueError(
                    "In forward pass, the output shape of "
                    f"Layer {name} "
                    "doesn't match the declared output shape:\n\t"
                    f"layer_expected_output.shape: {layer.output_shape}, "
                    f"layer_actual_output.shape: {output.shape[1:]}"
                )
        self._last_y_pred = output

        if verbose:
            total = sum(t for _, t in timings)
            print(f"Forward pass done in {total:.2f}s")
            for name, ftime in timings:
                print(f"{name:>20}: {ftime:>8.4f}s ({ftime / total:6>.2%})")
            print()

        return self._last_y_pred

    def backward(self, y_gt, verbose=False):
        if self._loss is None:
            raise ValueError("Loss is not defined")
        if self._last_y_pred.shape != y_gt.shape:
            raise ValueError(
                "Network output shape doesn't match ground truth shape:\n\t"
                f"output.shape: {self._last_y_pred.shape}, "
                f"y_gt.shape: {y_gt.shape}"
            )

        grad_outputs = self._loss.gradient(y_gt, self._last_y_pred)
        output_meaning = "the network output shape"
        timings = []

        for layer in self._layers[::-1]:
            name = layer.__class__.__name__
            if layer.output_shape != grad_outputs.shape[1:]:
                raise ValueError(
                    "In backward pass, the gradient of the output shape of "
                    f"Layer {name} "
                    f"doesn't match {output_meaning}:\n\t"
                    f"layer_expected_grad_output.shape: {layer.output_shape}, "
                    f"layer_actual_grad_output.shape: {grad_outputs.shape[1:]}"
                )
            output_meaning = "output shape of previous layer"

            backward_start = time.time()
            grad_outputs = layer.backward(grad_outputs)
            timings.append((name, time.time() - backward_start))

            if np.isnan(grad_outputs).any():
                raise ValueError(
                    "In backward pass, the gradient of the input of "
                    f"Layer {name} has NaN values."
                )

            if layer.input_shape != grad_outputs.shape[1:]:
                raise ValueError(
                    "In backward pass, the gradient of the input shape of "
                    f"Layer {name} "
                    "doesn't match the declared input shape:\n\t"
                    f"layer_expected_grad_input.shape: {layer.output_shape}, "
                    f"layer_actual_grad_input.shape: {grad_outputs.shape[1:]}"
                )

        if verbose:
            total = sum(t for _, t in timings)
            print(f"Backward pass done in {total:.2f}s")
            for name, btime in timings[::-1]:
                print(f"{name:>20}: {btime:>8.4f}s ({btime / total:6>.2%})")
            print()

    def fit_batch(self, x_batch, y_batch, verbose=False):
        if self._optimizer is None:
            raise ValueError("Optimizer is not defined")
        y_batch_pred = self.forward(x_batch, training=True, verbose=verbose)
        self.backward(y_batch, verbose=verbose)
        for layer in self._layers[::-1]:
            layer.update_parameters()
        return self.get_metrics(y_batch, y_batch_pred)

    def fit(
        self, x_train, y_train, batch_size, epochs,
        shuffle=True, verbose=True,
        x_valid=None, y_valid=None
    ):
        size = x_train.shape[0]
        x_gt, y_gt = x_train[:], y_train[:]

        start_epoch = len(self.loss_train_history) + 1
        epochs_range, display, update, description = range_fn(
            start_epoch, start_epoch + epochs,
            total=epochs * (size // batch_size)
        )
        description('Training')
        for epoch in epochs_range:
            if shuffle:
                p = np.random.permutation(size)
                x_gt, y_gt = x_train[p], y_train[p]

            train_metrics = np.empty((size // batch_size, 2))
            for step in range(size // batch_size):
                ind_slice = slice(step * batch_size, (step + 1) * batch_size)
                train_metrics[step] = self.fit_batch(
                    x_gt[ind_slice], y_gt[ind_slice],
                    verbose=verbose > 1
                )
                update()
            train_loss, train_acc = np.mean(train_metrics, axis=0)

            metrics = [
                ("Epoch", f"{epoch: >3}"),
                ("train loss", f"{train_loss:#.6f}"),
                ("train accuracy", f"{train_acc:.2%}"),
            ]

            if (x_valid is not None) and (y_valid is not None):
                valid_loss, valid_acc = self.evaluate(
                    x_valid, y_valid, batch_size
                )
                metrics.extend(
                    [
                        ("validation loss", f"{valid_loss:#.6f}"),
                        ("validation accuracy", f"{valid_acc:.2%}"),
                    ]
                )
            else:
                valid_loss, valid_acc = float('nan'), float('nan')

            if verbose:
                display(
                    ', '.join(
                        f"{name}: {value}" for name, value in metrics
                    )
                )

            self.loss_valid_history.append(valid_loss)
            self.loss_train_history.append(train_loss)
            self.accuracy_valid_history.append(valid_acc)
            self.accuracy_train_history.append(train_acc)
        if verbose:
            print()

    def get_metrics(self, y_gt, y_pred):
        losses = self._loss.value(y_gt, y_pred)
        matches = np.argmax(y_gt, axis=-1) == np.argmax(y_pred, axis=-1)
        return np.mean(losses), np.mean(matches)

    def evaluate(self, x_gt, y_gt, batch_size):
        if self._loss is None:
            raise ValueError("Loss is not defined")
        if x_gt.shape[0] != y_gt.shape[0]:
            raise ValueError("x and y must have equal size")

        y_pred = np.empty(y_gt.shape)
        size = x_gt.shape[0]
        for step in range(size // batch_size + 1):
            ind_slice = slice(step * batch_size, (step + 1) * batch_size)

            x_sliced = x_gt[ind_slice]
            if x_sliced.shape[0] == 0:
                continue

            y_pred[ind_slice] = self.forward(x_gt[ind_slice], training=False)

        return self.get_metrics(y_gt, y_pred)

    def __str__(self):
        total_params = 0
        layer_strs = []
        for layer in self._layers:
            layer_strs.append(str(layer))
            total_params += layer.num_parameters()

        layer_strs += [
            '-' * len(layer_strs[-1]),
            _model_fstring.format(name="Total", params=total_params)
        ]
        return '\n'.join(layer_strs)


# endregion

# region Skip Connection Layer
class SkipConnection(Layer):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = layers

        if (
            not isinstance(layers, (list, tuple)) or
            not len(layers) or
            not all(isinstance(l, Layer) for l in layers)
        ):
            raise ValueError(
                "The 'layers' argument of the SkipConnection Layer "
                "must be a non-empty list of Layers."
            )

    def __str__(self):
        text = []
        for idx, layer in enumerate(self.layers):
            if len(self.layers) == 1:
                c = '>'
            elif idx == 0:
                c = '\\'
            elif idx == len(self.layers) - 1:
                c = '/'
            else:
                c = '|'
            text.append(str(layer) + f' {c} Skip Connection')
        return '\n'.join(text)

    def num_parameters(self):
        return sum(layer.num_parameters() for layer in self.layers)

    def build(self, optimizer, prev_layer=None):
        super().build(optimizer, prev_layer)

        for layer in self.layers:
            layer.build(optimizer, prev_layer=prev_layer)
            prev_layer = layer

        if self.input_shape != self.layers[-1].output_shape:
            raise RuntimeError(
                "The input shape of the SkipConnection Layer and the output "
                "shape of the last skipped layer must match.\n\t"
                f"input_shape.shape: {self.input_shape}, "
                f"last_output_shape.shape: {self.layers[-1].output_shape}"
            )

    def update_parameters(self):
        for layer in self.layers:
            layer.update_parameters()

    def forward_impl(self, inputs):
        skip = inputs
        for layer in self.layers:
            layer.is_training = self.is_training
            inputs = layer(inputs)
        return skip + inputs

    def backward_impl(self, grad_outputs):
        skip = grad_outputs
        for layer in self.layers[::-1]:
            grad_outputs = layer.backward(grad_outputs)
        return skip + grad_outputs


# endregion

# region PyTorch convolve implementation
try:
    import torch as _torch
except ImportError:
    _torch_exception = e
    _torch = None


def _check_pytorch_installed():
    __tracebackhide__ = True
    if _torch is None:
        # Pretend, that the import error was raised by this function
        # instead of when the lazy import failed in global scope
        try:
            raise Exception()
        except Exception as e:
            _torch_exception.__traceback__ = e.__traceback__
            _torch_exception.__context__ = e.__context__

        # Re-raise the import exception
        raise _torch_exception


def _numpy_to_torch(V):
    if any(s < 0 for s in V.strides):
        # PyTorch doesn't currently support tensors with negative stride.
        # We can still avoid the copy, if all the strides are positive.
        V = V.copy()
    return _torch.from_numpy(V)


def convolve_pytorch(inputs, kernels, padding):
    _check_pytorch_installed()

    # Flip the kernel (pytorch implements cross-correlation, not convolution).
    kernels = kernels[:, :, ::-1, ::-1]

    with warnings.catch_warnings():
        # We set the inputs/kernels tensors as read-only so that the students
        # don't accidentally change them in their code.
        # PyTorch doesn't have non-writeable tensors, but we don't really care
        # since nn.functional.conv2d obviously doesn't write to its inputs.
        # So we just ignore the annoying warning.
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"The given NumPy array is not writeable, and PyTorch does "
                    r"not support non-writeable tensors\..*"
        )

        return _torch.nn.functional.conv2d(
            _numpy_to_torch(inputs),
            _numpy_to_torch(kernels),
            padding=padding
        ).numpy()

# endregion
