import tensorflow as tf
from gpflow.base import Parameter
from gpflow.utilities import positive

float_type = tf.float64
print("hellooo")
class Param:
    """
    A drop-in replacement for your old GPflow 1.x Param + transforms.
    If `fixed=True`, returns a constant; otherwise a GPflow Parameter
    constrained by the supplied bijector (default: identity).
    """
    def __init__(self,
                 value,
                 transform=None,   # either None, the string "positive", or a TFP bijector
                 fixed=False,
                 name=None,
                 learning_rate=None,
                 summ=False):
        self.value = value
        self.fixed = fixed
        self.name = name or "param"

        # pick the bijector
        if transform is None:
            bijector = None
        elif transform == "positive":
            bijector = positive()    # softplus(·), same as old Log1pe
        else:
            bijector = transform     # assume you passed in a tfp.bijectors.Bijector

        if fixed:
            # just a constant tensor
            self._var = tf.constant(value, dtype=float_type, name=self.name)
        else:
            # a GPflow Parameter = a TF variable + bijector
            self._var = Parameter(value,
                                  transform=bijector,
                                  name=self.name)
        if summ:
            tf.summary.histogram(self.name, self._var)

    def __call__(self):
        # always return the *constrained* value
        return self._var

    def assign(self, new_value):
        """
        If you ever need to overwrite it:
        """
        if self.fixed:
            raise RuntimeError("Param is fixed; cannot assign.")
        self._var.assign(new_value)

    @property
    def shape(self):
        return tf.shape(self.value)
