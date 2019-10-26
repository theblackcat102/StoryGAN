import os
import tqdm
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import random
import re
import pdb
from PIL import Image, ImageSequence

def read_gif_file(filename, seek_pos=1):
	img = Image.open(filename)
	try:
		img.seek(seek_pos)
	except EOFError:
		return img, 1
	return img, seek_pos

class StoryDataset(torch.utils.data.Dataset):
	def __init__(self, image_path, transform, videl_len = 4, is_train = True, 
		gif_path='/mnt/workstation/Scenes_Dialogues',
		cache_path='./.cache', cache_name='data_cache.pt'):
		self.dir_path = image_path
		self.descriptions = torch.load(os.path.join(cache_path, cache_name))
		self.transforms = transform
		self.video_len = 5
		self.gif_path = gif_path

	def __getitem__(self, item):
		flow = self.descriptions[item]

		super_label = []
		image = []
		lists = []
		des = []

		for idx in range(len(flow)):
			sent = flow[idx]
			# sample_frame = random.uniform(0, flow['frame_count'])
			gif_file = os.path.join(self.gif_path, sent['img']+'.gif')
			im, seek_pos = read_gif_file(gif_file, seek_pos=1)

			image.append( np.expand_dims(np.array(im.convert("RGB")), axis = 0) )
			sent_embed = sent['sent_embed'][0]
			des.append(np.expand_dims(sent_embed, axis = 0))
			super_label.append(np.asarray(sent['label']))

		super_label[0] = np.expand_dims(super_label[0], axis = 0)
		for i in range(1, self.video_len):
			super_label[i] = super_label[i] + super_label[i-1]
			super_temp = super_label[i].reshape(-1)
			super_temp[super_temp>1] = 1
			super_label[i] = np.expand_dims(super_temp, axis = 0)

		des = np.concatenate(des, axis = 0)
		image_numpy = np.concatenate(image, axis = 0)
		super_label = np.concatenate(super_label, axis = 0)
		# image is T x H x W x C
		image = self.transforms(image_numpy)
		# After transform, image is C x T x H x W
		des = torch.tensor(des)
		## des is attribute, subs is encoded text description
		return {'images': image, 'description': des, 'label':super_label}

	def __len__(self):
		return len(self.descriptions)


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, image_path, transform,   
		video_len = 4, is_train = True, cache_path='./.cache', cache_name='data_cache.pt',
		gif_path='/mnt/workstation/Scenes_Dialogues',):
		self.dir_path = image_path
		self.descriptions = torch.load(os.path.join(cache_path, cache_name))
		self.transforms = transform
		self.video_len = 5
		self.gif_path = gif_path


	def __getitem__(self, item):
		se = np.random.randint(0,self.video_len, 1)[0]
		sent = self.descriptions[item][se]

		gif_file = os.path.join(self.gif_path, sent['img']+'.gif')
		# sample_frame = random.uniform(0, sent['frame_count'])
		im, seek_pos = read_gif_file(gif_file, seek_pos=1)

		image = np.array(im.convert("RGB"))

		image = self.transforms(image)

		des = random.choice(sent['sent_embed'])
		super_label = np.asarray(sent['label'])


		content = []
		flow = self.descriptions[item]
		for idx in range(len(flow)):
			sent = flow[idx]
			sent_embed = sent['sent_embed'][0]
			content.append(np.expand_dims(sent_embed.astype(np.float32), axis = 0))

		super_label = super_label.reshape(-1)
		super_label[super_label>1] = 1
		content = np.concatenate(content, 0)
		content = torch.tensor(content)
		## des is attribute, subs is encoded text description
		return {'images': image, 'description': des, 'label':super_label, 'content': content}

	def __len__(self):
		return len(self.descriptions)


def video_transform(video, image_transform):
	vid = []
	for im in video:
		vid.append(image_transform(im))

	vid = torch.stack(vid).permute(1, 0, 2, 3)

	return vid

if __name__ == "__main__":
	n_channels = 3

	image_transforms = transforms.Compose([
		PIL.Image.fromarray,
		transforms.Resize( (64, 64) ),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		lambda x: x[:n_channels, ::],
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	video_transforms = functools.partial(video_transform, image_transform=image_transforms)

	dataset = ImageDataset('/mnt/workstation/Scenes_Dialogues', image_transforms)
	print(dataset[0]['images'].shape)

