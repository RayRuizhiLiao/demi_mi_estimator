import os

import torch
import torch.nn as nn

from utils import make_mlp


class Discriminator(nn.Module):
	"""A neural network discriminator.
	"""
	def __init__(self, sample_length, hidden_dims):
		super(Discriminator, self).__init__()

		self.sample_length = sample_length
		self.hidden_dims = hidden_dims
		self.mlp = make_mlp(2*sample_length, hidden_dims)

	def forward(self, x):
		x = torch.flatten(x, start_dim=1)
		x = torch.reshape(x, (x.size()[0], 1, x.size()[-1]))
		logit = self.mlp(x)
		
		return logit

	def save_pretrained(self, save_directory):
		""" Save the discriminator with its configuration file to a directory, 
			so that it can be re-loaded 
			using the `from_pretrained(save_directory)` class method.
		"""
		assert os.path.isdir(save_directory), \
			"Saving path should be a directory where the model and configuration can be saved"

		# Only save the model it-self if we are using distributed training
		model_to_save = self.module if hasattr(self, 'module') else self

		# If we save using the predefined names, we can load using `from_pretrained`
		output_model_file = os.path.join(save_directory, 'pytorch_discriminator.bin')
		torch.save(model_to_save.state_dict(), output_model_file)

	@classmethod
	def from_pretrained(cls, pretrained_model_path, **kwargs):
		""" Re-load a trained model.
		"""		
		sample_length = kwargs.pop('sample_length', 20)
		hidden_dims = kwargs.pop('hidden_dims', [256,256])

		# Specify the pretrained discriminator path
		if os.path.isdir(pretrained_model_path):
			archive_file = os.path.join(pretrained_model_path, 'pytorch_discriminator.bin')
		else:
			raise Exception('Please provide a directory to load the model from, currently given',
				pretrained_model_path)

		#print("Loading the discriminator model")
		# Instantiate the model
		model = cls(sample_length=sample_length, 
					hidden_dims=hidden_dims)
		#print("Instantiated model summary: {}".format(model))

		state_dict = torch.load(archive_file, map_location='cpu')

		# Convert old format to new format if needed from a PyTorch state_dict
		old_keys = []
		new_keys = []
		for key in state_dict.keys():
			new_key = None
			if 'gamma' in key:
				new_key = key.replace('gamma', 'weight')
			if 'beta' in key:
				new_key = key.replace('beta', 'bias')
			if new_key:
				old_keys.append(key)
				new_keys.append(new_key)
		for old_key, new_key in zip(old_keys, new_keys):
			state_dict[new_key] = state_dict.pop(old_key)

		# Load from a PyTorch state_dict
		missing_keys = []
		unexpected_keys = []
		error_msgs = []
		# copy state_dict so _load_from_state_dict can modify it
		metadata = getattr(state_dict, '_metadata', None)
		state_dict = state_dict.copy()
		if metadata is not None:
			state_dict._metadata = metadata

		def load(module, prefix=''):
			local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
			module._load_from_state_dict(
				state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
			for name, child in module._modules.items():
				if child is not None:
					load(child, prefix + name + '.')

		load(model)
		if len(missing_keys) > 0:
			logger.info("Weights of {} not initialized from pretrained model: {}".format(
				model.__class__.__name__, missing_keys))
		if len(unexpected_keys) > 0:
			logger.info("Weights from pretrained model not used in {}: {}".format(
				model.__class__.__name__, unexpected_keys))
		if len(error_msgs) > 0:
			raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
								model.__class__.__name__, "\n\t".join(error_msgs)))

		return model