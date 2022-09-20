#retrieved from here: https://github.com/cpury/keras_gradient_noise/blob/master/keras_gradient_noise/gradient_noise.py
# on 28 Sept 2021
# modified: dknox 28 Sept 2021 to only use tensorflow keras
import inspect
import importlib

def add_gradient_noise(BaseOptimizer, keras=None):
	"""
	Given a Keras-compatible optimizer class, returns a modified class that
	supports adding gradient noise as introduced in this paper:
	https://arxiv.org/abs/1511.06807
	The relevant parameters from equation 1 in the paper can be set via
	noise_eta and noise_gamma, set by default to 0.3 and 0.55 respectively.
	By default, tries to guess whether to use default Keras or tf.keras based
	on where the optimizer was imported from. You can also specify which Keras
	to use by passing the imported module.
	"""
	if keras is None:
		# Import it automatically. Try to guess from the optimizer's module
		keras = importlib.import_module('tensorflow.keras')

	K = keras.backend

	if not (
		inspect.isclass(BaseOptimizer) and
		issubclass(BaseOptimizer, keras.optimizers.Optimizer)
	):
		raise ValueError(
			'add_gradient_noise() expects a valid Keras optimizer'
		)

	def _get_shape(x):
		if hasattr(x, 'dense_shape'):
			return x.dense_shape

		return K.shape(x)

	class NoisyOptimizer(BaseOptimizer):
		def __init__(self, noise_eta=0.3, noise_gamma=0.55, **kwargs):
			super(NoisyOptimizer, self).__init__(**kwargs)
			with K.name_scope(self.__class__.__name__):
				self.noise_eta = K.variable(noise_eta, name='noise_eta')
				self.noise_gamma = K.variable(noise_gamma, name='noise_gamma')

		def get_gradients(self, loss, params):
			grads = super(NoisyOptimizer, self).get_gradients(loss, params)

			# Add decayed gaussian noise
			t = K.cast(self.iterations, K.dtype(grads[0]))
			variance = self.noise_eta / ((1 + t) ** self.noise_gamma)

			grads = [
				grad + K.random_normal(
					_get_shape(grad),
					mean=0.0,
					stddev=K.sqrt(variance),
					dtype=K.dtype(grads[0])
				)
				for grad in grads
			]

			return grads

		def get_config(self):
			config = {'noise_eta': float(K.get_value(self.noise_eta)),
					  'noise_gamma': float(K.get_value(self.noise_gamma))}
			base_config = super(NoisyOptimizer, self).get_config()
			return dict(list(base_config.items()) + list(config.items()))

	NoisyOptimizer.__name__ = 'Noisy{}'.format(BaseOptimizer.__name__)

	return NoisyOptimizer

