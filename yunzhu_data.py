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
		
		lat = 'latin1'
		textvec = '/mnt/HDD/pororoSV/'
		self.descriptions = np.load(textvec+'descriptions_vec.npy', allow_pickle=True,encoding=lat).item()
		self.attributes =  np.load(textvec+'descriptions_attr.npy',allow_pickle=True,encoding=lat).item()
		self.subtitles = np.load(textvec+'subtitles_vec.npy',allow_pickle=True,encoding=lat).item()
		self.labels = np.load(textvec+'labels.npy',allow_pickle=True,encoding=lat).item()

		self.base = torch.load(os.path.join(cache_path, cache_name))

		self.transforms = transform
		self.video_len = 5
		self.gif_path = gif_path

	def __getitem__(self, item):
		flow = self.base[item]

		super_label = []
		image = []
		lists = []
		des = []
		labels = []
		attributes = []



		for idx in range(len(flow)):
			sent = flow[idx]
			# sample_frame = random.uniform(0, flow['frame_count'])
			gif_file = os.path.join(self.gif_path, sent['img']+'.gif')
			im, seek_pos = read_gif_file(gif_file, seek_pos=1)

			image.append( np.expand_dims(np.array(im.convert("RGB")), axis = 0) )

			key = sent['img']#.decode('utf-8')
			se = np.random.randint(0,len(self.descriptions[key]),1)[0]
			sent_embed = np.expand_dims(self.descriptions[key][se].astype(np.float32), axis = 0)

			# des = self.descriptions[key][se]
			attri = self.attributes[key][se].astype('float32')
			label = self.labels[key].astype(np.float32)

			labels.append(np.expand_dims(label, axis=0))
			attributes.append(np.expand_dims(attri, axis = 0))
			des.append(sent_embed)
			super_label.append(np.asarray(sent['label']).astype('float32'))

		super_label[0] = np.expand_dims(super_label[0], axis = 0)
		for i in range(1, self.video_len):
			super_label[i] = super_label[i] + super_label[i-1]
			super_temp = super_label[i].reshape(-1)
			super_temp[super_temp>1] = 1
			super_label[i] = np.expand_dims(super_temp, axis = 0)

		attributes = np.concatenate(attributes, axis = 0)
		des = np.concatenate(des, axis = 0)
		labels = np.concatenate(labels, axis=0)
		des = np.concatenate([des, attributes], axis = 1)
		image_numpy = np.concatenate(image, axis = 0)
		super_label = np.concatenate(super_label, axis = 0)
		# image is T x H x W x C
		# print(image_numpy.shape)
		image = self.transforms(image_numpy)
		# After transform, image is C x T x H x W
		des = torch.tensor(des)
		## des is attribute, subs is encoded text description
		return {'images': image, 'description': des, 'label': labels}

	def __len__(self):
		return len(self.base)


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, image_path, transform,
		video_len = 4, is_train = True, cache_path='./.cache', cache_name='data_cache.pt',
		gif_path='/mnt/workstation/Scenes_Dialogues',):
		self.dir_path = image_path
		textvec = '/mnt/HDD/pororoSV/'
		lat = 'latin1'
		self.descriptions = np.load(textvec+'descriptions_vec.npy', allow_pickle=True,encoding=lat).item()
		self.attributes =  np.load(textvec+'descriptions_attr.npy',allow_pickle=True,encoding=lat).item()
		self.subtitles = np.load(textvec+'subtitles_vec.npy',allow_pickle=True,encoding=lat).item()
		self.descriptions_original = np.load(textvec+'descriptions.npy',allow_pickle=True,encoding=lat).item()
		self.labels = np.load(textvec+'labels.npy',allow_pickle=True,encoding=lat).item()

		self.base = torch.load(os.path.join(cache_path, cache_name))
		self.transforms = transform
		self.video_len = 5
		self.gif_path = gif_path


	def __getitem__(self, item):
		se = np.random.randint(0,self.video_len, 1)[0]
		sent = self.base[item][se]

		gif_file = os.path.join(self.gif_path, sent['img']+'.gif')
		key = sent['img']#.decode('utf-8')
		sample_frame = int(random.uniform(0, sent['frame_count']))
		im, seek_pos = read_gif_file(gif_file, seek_pos=sample_frame)

		image = np.array(im.convert("RGB"))

		image = self.transforms(image)
		se = np.random.randint(0,len(self.descriptions[key]),1)[0]

		attri = self.attributes[key][se].astype('float32')
		attri = np.expand_dims(attri.astype('float32'), axis = 0)

		des = np.expand_dims(self.descriptions[key][se].astype(np.float32), axis = 0)
		des = np.concatenate([des, attri], axis=1)
		super_label = np.asarray(sent['label'])

		content = []
		attris = []
		attribute_label = []
		flow = self.base[item]
		for idx in range(len(flow)):
			sent = flow[idx]
			key = sent['img']#.decode('utf-8')
			se = np.random.randint(0,len(self.descriptions[key]),1)[0]

			sent_embed = np.expand_dims(self.descriptions[key][se].astype(np.float32), axis = 0)

			attri = self.attributes[key][se].astype('float32')
			attris.append(np.expand_dims(attri.astype('float32'), axis = 0))

			label = self.labels[key].astype(np.float32)
			attribute_label.append(np.expand_dims(label, axis=0))
			content.append(sent_embed)

		super_label = super_label.reshape(-1)
		super_label[super_label>1] = 1

		content = np.concatenate(content, axis = 0)
		attris = np.concatenate(attris, axis = 0)
		attribute_label = np.concatenate(attribute_label, axis = 0)

		content = np.concatenate([content, attris, attribute_label  ], 1)
		content = torch.tensor(content)
		## des is attribute, subs is encoded text description
		return {'images': image, 'description': des, 'label':super_label, 'content': content}

	def __len__(self):
		return len(self.base)

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

	dataset = StoryDataset('/mnt/workstation/Scenes_Dialogues', video_transforms)
	print(dataset[0]['description'].shape)

