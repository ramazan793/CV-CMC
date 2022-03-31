from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            return parameter - parameter_grad * self.lr

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            updater.inertia = updater.inertia * self.momentum + self.lr * parameter_grad
            return parameter - updater.inertia

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        forward = np.copy(inputs)
        forward[forward < 0] = 0
        return forward

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        f_grad = np.copy(self.forward_inputs)
        f_grad[f_grad >= 0] = 1
        f_grad[f_grad < 0] = 0
        
        return f_grad * grad_outputs 


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        normalized_inputs = inputs - np.max(inputs, axis = 1)[:, None]
        forward = np.exp(normalized_inputs) / np.sum(np.exp(normalized_inputs), axis = 1)[:, None]
        return forward

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs
    
            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        y = self.forward_outputs
        return grad_outputs * y - np.sum(y * grad_outputs, axis = 1)[:,None] * y


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        return inputs @ self.weights + self.biases
    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        
        self.weights_grad = self.forward_inputs.T @ grad_outputs / grad_outputs.shape[0]
        self.biases_grad = np.mean(grad_outputs, axis = 0)
        
        
        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n,)), loss scalars for batch

                n - batch size
                d - number of units
        """
        return -np.log(y_pred[y_gt != 0])

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), gradient loss to y_pred

                n - batch size
                d - number of units
        """
        nz = np.copy(y_pred)
        nz[y_pred <= eps] = eps
        return -1 / nz * y_gt


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):

    model = Model(CategoricalCrossentropy(), SGDMomentum(lr = 1e-3, momentum = 0.8))

    model.add(Dense(input_shape = (784,), units = 512))
    model.add(ReLU())
    model.add(Dense(units = 128))
    model.add(ReLU())
    
    model.add(Dense(units = 10))
    model.add(Softmax())
    
    print(model)

    model.fit(x_train, y_train, batch_size = 64, epochs = 5, x_valid = x_valid, y_valid = y_valid)

    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    n, d, ih, iw = inputs.shape
    c, d, kh, kw = kernels.shape
    
    kernels = np.flip(kernels, axis = (-2, -1))
    
    oh, ow = ih - kh + 1 + 2*padding, iw - kw + 1 + 2*padding
    out = np.zeros((n, c, oh, ow), dtype = inputs.dtype)
    
    padded_inputs = np.zeros((n, d, ih + 2*padding, iw + 2*padding))
    padded_inputs[:, :, padding:ih + padding, padding:iw + padding] = inputs
    
    for i in range(oh):
        for j in range(ow):
            a = kernels * padded_inputs[:, :, i:i + kh, j:j + kw][:, None, :, :, :]
            out[:, :, i , j] = np.sum(a, axis = (-1, -2, -3), keepdims = False).reshape((n,c))

    return out


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        padding = (self.kernel_size - 1) // 2 # always 'same' padding
        return convolve(inputs, self.kernels, padding) + self.biases[None, :, None, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        def transpose(X):
            return np.transpose(X, (1, 0, 2, 3))
        
        d = self.kernels_grad.shape[1]
        n, c, h, w = grad_outputs.shape
        padding = (self.kernel_size - 1) // 2
        
        X = np.flip(self.forward_inputs, axis = (-2, -1))
        
        self.kernels_grad = transpose(convolve(transpose(X), transpose(grad_outputs), padding)) / n
        self.biases_grad = np.mean(np.sum(grad_outputs, axis=(-2,-1)), axis = 0)
                
        K = np.flip(self.kernels, axis = (-2, -1))
        inv_padding = self.kernel_size - padding - 1
        
        inputs_grad = convolve(grad_outputs, transpose(K), inv_padding)
        
        return inputs_grad


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        n, d, w, h = inputs.shape
        
        if self.pool_mode == 'max':
            reduce = np.max
        else:
            reduce = np.mean

        r = np.lib.stride_tricks.sliding_window_view(inputs, (self.pool_size, self.pool_size), axis = (-2,-1))
        r = r[:, :, ::self.pool_size, ::self.pool_size]
        
        # get windows
        mask_view = np.lib.stride_tricks.sliding_window_view(inputs, (self.pool_size, self.pool_size), axis = (-2,-1), writeable = True)
        mask_view = np.copy(mask_view)
        mask_view = mask_view[:, :, ::self.pool_size, ::self.pool_size]
        shape = mask_view.shape
        mask_view = np.reshape(mask_view, (n,d,-1,self.pool_size**2))
        
        # find argmax in each
        argmax_at_each_window = mask_view.argmax(axis=3)
        mask_view[...] = 0
        np.put_along_axis(mask_view, argmax_at_each_window[:,:,:,None], 1, axis=-1)
        
        # reshape correctly
        mask_view = mask_view.reshape(shape)
        mask_view = mask_view.swapaxes(-3,-2).reshape(n,d,w,h)
        
        self.forward_idxs = mask_view.astype(bool)
        
        return reduce(r, axis = (4, 5))

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        def unpool(p):
            return np.kron(p, np.ones((self.pool_size, self.pool_size), dtype=np.float64))
        n, d = grad_outputs.shape[:2]
        
        if self.pool_mode == 'max':
            grad = unpool(grad_outputs) * self.forward_idxs
        else:
            grad = unpool(grad_outputs) * 1 / self.pool_size**2
        return grad


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        mean = var = 0
        if self.is_training:
            mean = np.mean(inputs, axis = (0, -2, -1), keepdims = True)
            var = np.var(inputs, axis = (0, -2, -1), keepdims = True)

            self.forward_centered_inputs = inputs - mean
            self.forward_inverse_std = 1 / np.sqrt(eps + var.ravel())
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std[:, None, None]
            
            self.running_mean = self.momentum  * self.running_mean + (1 - self.momentum) * mean.ravel()
            self.running_var = self.momentum  * self.running_var + (1 - self.momentum) * var.ravel()
        else:
            self.forward_normalized_inputs = (inputs - self.running_mean[:, None, None]) / np.sqrt(eps + self.running_var[:, None, None])
        
        return self.forward_normalized_inputs * self.gamma[:, None, None] + self.beta[:, None, None]

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        self.beta_grad = np.sum(grad_outputs, axis = (0, 2, 3)) / grad_outputs.shape[0]
        self.gamma_grad = np.sum(self.forward_normalized_inputs * grad_outputs, axis = (0, 2, 3)) / grad_outputs.shape[0]
        
        n, d, h, w = grad_outputs.shape
        
        n = n*h*w
        dl_dn = grad_outputs * self.gamma[None, :, None, None]
        grad = self.forward_inverse_std[None, :, None, None] * (n * dl_dn - self.forward_normalized_inputs * np.sum(dl_dn * self.forward_normalized_inputs, axis = (0,2,3), keepdims = True) - np.sum(dl_dn, axis = (0,2,3), keepdims = True)) / n
        return grad

# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        return np.reshape(inputs, (inputs.shape[0], -1))

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        return grad_outputs.reshape(grad_outputs.shape[0], *self.input_shape)


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        if self.is_training:
            self.forward_mask = np.random.uniform(0, 1, inputs.shape) > self.p
            return self.forward_mask * inputs
        else:
            return inputs * (1 - self.p)

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        return self.forward_mask * grad_outputs
    
# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr = 1e-3, momentum = 0.9))
    
    model.add(Conv2D(32, input_shape = (3, 32, 32)))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())
    
    model.add(Conv2D(64))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())
    
    model.add(Conv2D(128))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D())
    
    model.add(Flatten())
              
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dense(10))
    
    model.add(Softmax())
    
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size = 64, epochs = 5, x_valid = x_valid, y_valid = y_valid)

    return model

# ============================================================================
