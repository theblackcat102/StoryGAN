from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
import torch.optim as optim
import os
import time
from tqdm import tqdm
import pdb
import numpy as np
import torchfile
from tensorboardX import SummaryWriter
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_story_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss, save_test_samples

from shutil import copyfile

class GANTrainer(object):
	def __init__(self, output_dir, ratio = 1.0, test_dir = None, tensorboard=False):
		self.model_dir = os.path.join(output_dir, 'Model')
		self.image_dir = os.path.join(output_dir, 'Image')
		self.log_dir = os.path.join(output_dir, 'Log')
		mkdir_p(self.model_dir)
		mkdir_p(self.image_dir)
		mkdir_p(self.log_dir)
		self.video_len = cfg.VIDEO_LEN
		self.max_epoch = cfg.TRAIN.MAX_EPOCH
		self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
		self.test_dir = test_dir
		s_gpus = cfg.GPU_ID.split(',')
		self.gpus = [int(ix) for ix in s_gpus]
		self.num_gpus = len(self.gpus)
		self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
		self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
		self.ratio = ratio
		self.tensorboard = tensorboard
		self.writer = None
		torch.cuda.set_device(self.gpus[0])
		cudnn.benchmark = True

	# ############# For training stageI GAN #############
	def load_networks(self):
		from model import StoryGAN, D_IMG, D_STY
		netG = StoryGAN(self.video_len)
		netG.apply(weights_init)
		print(netG)
		netD_im = D_IMG()
		netD_im.apply(weights_init)
		netD_st = D_STY()
		netD_st.apply(weights_init)
		print(netD_im)
		print(netD_st)

		if cfg.NET_G != '':
			state_dict = \
				torch.load(cfg.NET_G,
						   map_location=lambda storage, loc: storage)
			netG.load_state_dict(state_dict)
			print('Load from: ', cfg.NET_G)
		if cfg.NET_D != '':
			state_dict = \
				torch.load(cfg.NET_D,
						   map_location=lambda storage, loc: storage)
			netD.load_state_dict(state_dict)
			print('Load from: ', cfg.NET_D)
		if cfg.CUDA:
			netG.cuda()
			netD_im.cuda()
			netD_st.cuda()
		return netG, netD_im, netD_st


	def sample_real_image_batch(self):
		if self.imagedataset is None:
			self.imagedataset = enumerate(self.imageloader)
		batch_idx, batch = next(self.imagedataset)
		b = batch
		if cfg.CUDA:
			for k, v in batch.items():
				if k == 'text':
					continue
				else:
					b[k] = v.cuda()

		if batch_idx == len(self.imageloader) - 1:
			self.imagedataset = enumerate(self.imageloader)
		return b


	def build_tensorboard(self):
		model_name = str(datetime.datetime.now().strftime("story_%b-%d_%H-%M-%S"))
		print('save stats to tensorboard...')
		tensorboard_log_dir = './logs'
		tensorboard_log_dir = os.path.join(tensorboard_log_dir, model_name)
		writer = SummaryWriter(tensorboard_log_dir,
			comment="Story GAN")
		self.writer = writer

	def train(self, imageloader, storyloader, testloader):
		self.imageloader = imageloader
		self.testloader = testloader
		self.imagedataset = None
		self.testdataset = None
		netG, netD_im, netD_st = self.load_networks()


		im_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
		im_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
		st_real_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(1))
		st_fake_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(0))
		if cfg.CUDA:
			im_real_labels, im_fake_labels = im_real_labels.cuda(), im_fake_labels.cuda()
			st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()

		generator_lr = cfg.TRAIN.GENERATOR_LR
		discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR

		lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
		im_optimizerD = \
			optim.Adam(netD_im.parameters(),
					   lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

		st_optimizerD = \
			optim.Adam(netD_st.parameters(),
					   lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

		netG_para = []
		for p in netG.parameters():
			if p.requires_grad:
				netG_para.append(p)
		optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.GENERATOR_LR,
								betas=(0.5, 0.999))
		if self.tensorboard:
			self.build_tensorboard()
		loss = {}
		step = 0
		for epoch in range(self.max_epoch):
			start_t = time.time()
			if epoch % lr_decay_step == 0 and epoch > 0:
				generator_lr *= 0.5
				for param_group in optimizerG.param_groups:
					param_group['lr'] = generator_lr
				discriminator_lr *= 0.5
				for param_group in st_optimizerD.param_groups:
					param_group['lr'] = discriminator_lr
				for param_group in im_optimizerD.param_groups:
					param_group['lr'] = discriminator_lr
				loss.update({
					'D/lr': discriminator_lr,
					'G/lr': generator_lr,
				})

			print('Epoch [{}/{}]:'.format(epoch, self.max_epoch))
			with tqdm(total=len(storyloader), dynamic_ncols=True) as pbar:
				for i, data in enumerate(storyloader, 0):
					######################################################
					# (1) Prepare training data
					######################################################
					im_batch = self.sample_real_image_batch()
					st_batch = data

					im_real_cpu = im_batch['images']
					im_motion_input = im_batch['description']
					im_content_input = im_batch['content']
					im_content_input = im_content_input.mean(1).squeeze()
					im_catelabel = im_batch['label']
					im_real_imgs = Variable(im_real_cpu)
					im_motion_input = Variable(im_motion_input)
					im_content_input = Variable(im_content_input)

					st_real_cpu = st_batch['images']
					st_motion_input = st_batch['description']
					st_content_input = st_batch['description']
					st_catelabel = st_batch['label']
					st_real_imgs = Variable(st_real_cpu)
					st_motion_input = Variable(st_motion_input)
					st_content_input = Variable(st_content_input)

					if cfg.CUDA:
						st_real_imgs = st_real_imgs.cuda()
						im_real_imgs = im_real_imgs.cuda()
						st_motion_input = st_motion_input.cuda()
						im_motion_input = im_motion_input.cuda()
						st_content_input = st_content_input.cuda()
						im_content_input = im_content_input.cuda()
						im_catelabel = im_catelabel.cuda()
						st_catelabel = st_catelabel.cuda()
					#######################################################
					# (2) Generate fake stories and images
					######################################################

					im_inputs = (im_motion_input, im_content_input)
					_, im_fake, im_mu, im_logvar = netG.sample_images(*im_inputs)

					st_inputs = (st_motion_input, st_content_input)
					_, st_fake, c_mu, c_logvar, m_mu, m_logvar = netG.sample_videos(*st_inputs)


					############################
					# (3) Update D network
					###########################
					netD_im.zero_grad()
					netD_st.zero_grad()

					im_errD, im_errD_real, im_errD_wrong, im_errD_fake, accD = \
						compute_discriminator_loss(netD_im, im_real_imgs, im_fake,
												im_real_labels, im_fake_labels, im_catelabel,
												im_mu, self.gpus)

					st_errD, st_errD_real, st_errD_wrong, st_errD_fake, _ = \
						compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
												st_real_labels, st_fake_labels, st_catelabel,
												c_mu, self.gpus)

					loss.update({
						'D/story/loss': st_errD.data,
						'D/story/real_loss': st_errD_real.data,
						'D/story/fake_loss': st_errD_fake.data,
						'D/image/accuracy': accD,
						'D/image/loss': im_errD.data,
						'D/image/real_loss': im_errD_real.data,
						'D/image/fake_loss': im_errD_fake.data,
					})

					im_errD.backward()
					st_errD.backward()

					im_optimizerD.step()
					st_optimizerD.step()


					############################
					# (2) Update G network
					###########################
					for g_iter in range(2):
						netG.zero_grad()

						_, st_fake, c_mu, c_logvar, m_mu, m_logvar = netG.sample_videos(
							st_motion_input, st_content_input)

						_, im_fake, im_mu, im_logvar = netG.sample_images(im_motion_input, im_content_input)

						im_errG, accG = compute_generator_loss(netD_im, im_fake,
													im_real_labels, im_catelabel, im_mu, self.gpus)
						st_errG, _ = compute_generator_loss(netD_st, st_fake,
													st_real_labels, st_catelabel, c_mu, self.gpus)
						im_kl_loss = KL_loss(im_mu, im_logvar)
						st_kl_loss = KL_loss(m_mu, m_logvar)
						errG = im_errG + self.ratio * st_errG

						kl_loss = im_kl_loss + self.ratio * st_kl_loss
						loss.update({
							'G/loss': im_errG.data,
							'G/kl': kl_loss.data,
						})
						errG_total = im_errG + self.ratio * st_errG + kl_loss
						errG_total.backward()
						optimizerG.step()
					if self.writer:
						for key, value in loss.items():
							self.writer.add_scalar(key,  value,  step)

					step += 1
					pbar.update(1)

					if i % 100 == 0:
						# save the image result for each epoch
						lr_fake, fake, _, _, _, _ = netG.sample_videos(st_motion_input, st_content_input)
						save_story_results(st_real_cpu, fake, epoch, self.image_dir, writer=self.writer, steps=step)
						if lr_fake is not None:
							save_story_results(None, lr_fake, epoch, self.image_dir, writer=self.writer, steps=step)

			end_t = time.time()
			print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
					 Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
					 accG: %.4f accD: %.4f
					 Total Time: %.2fsec
				  '''
				  % (epoch, self.max_epoch, i, len(storyloader),
					 st_errD.data, st_errG.data,
					 st_errD_real, st_errD_wrong, st_errD_fake, accG, accD,
					 (end_t - start_t)))

			if epoch % self.snapshot_interval == 0:
				save_model(netG, netD_im, netD_st, epoch, self.model_dir)
				save_test_samples(netG, self.testloader, self.test_dir, writer=self.writer, steps=step)
		#
		save_model(netG, netD_im, netD_st, self.max_epoch, self.model_dir)
	#