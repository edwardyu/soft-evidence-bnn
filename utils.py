import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


def load_labelme(data_dir, dataset_name, model):
	train_set = dsets.ImageFolder(data_dir + '/' + dataset_name + '/train/', transform=transforms.Compose([
			transforms.Resize(255),
			transforms.RandomCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
		]))
	answers = []
	with open(data_dir+'/'+dataset_name+'/answers.txt', 'rb') as f:
		for line in f:
			answers.append([int(e) for e in line.strip().decode('utf-8').split(' ')])   

	# convert to categorical labels
	categorical_labels = []
	for row in answers:
		categorical = np.zeros(8)
		for vote in row:
			if vote != -1:
				categorical[vote] += 1
		categorical_labels.append(categorical / np.sum(categorical))

	categorical_labels_train = np.array(categorical_labels)
	filenames_train = []
	with open(data_dir+'/'+dataset_name+'/filenames_train.txt', 'rb') as f:
		lines = f.readlines()
		for l in lines:
			filenames_train.append(l.strip().decode('utf-8'))

	new_samples = []
	for i, (d, s) in enumerate(train_set.samples):
		filename = d.split('/')[-1]
		idx = filenames_train.index(filename)
		new_samples.append((d, categorical_labels_train[idx]))
	train_set.samples = new_samples
	train_loader = torch.utils.data.DataLoader(
		train_set,
		batch_size=8, shuffle=True,
		num_workers=4, pin_memory=True)
	test_set = dsets.ImageFolder('data/LabelMe/test/', transform=transforms.Compose([
			transforms.Resize(255),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
		]))
	labels_test = []
	with open(data_dir+'/'+dataset_name+'/labels_test.txt', 'rb') as f:
		for line in f:
			labels_test.append(int(line.strip().decode('utf-8'))) 
	categorical_labels_test = []
	for label in labels_test:
		categorical = np.zeros(8)
		categorical[label] = 1
		categorical_labels_test.append(categorical)
	categorical_labels_test = np.array(categorical_labels_test)
	filenames_test = []
	with open(data_dir+'/'+dataset_name+'/filenames_test.txt', 'rb') as f:
		lines = f.readlines()
		for l in lines:
			filenames_test.append(l.strip().decode('utf-8'))

	new_samples2 = []
	for i, (d, s) in enumerate(test_set.samples):
		filename = d.split('/')[-1]
		idx = filenames_test.index(filename)
		new_samples2.append((d, categorical_labels_test[idx]))
	test_set.samples = new_samples2
	test_loader = torch.utils.data.DataLoader(
		test_set,
		batch_size=8, shuffle=True,
		num_workers=4, pin_memory=True)	
	return train_loader, test_loader


def load_cifar(data_dir, dataset_name, model):
	full_data =	dsets.CIFAR10(download=True, root=data_dir, train=False, transform=transforms.Compose([
		transforms.ToTensor(),
	]))
	categorical_labels = np.load(data_dir + '/' + dataset_name + '/cifar10h-probs.npy')
	full_data.targets = categorical_labels
	train_data, test_data = torch.utils.data.random_split(full_data, [7000, 3000])
	train_loader = torch.utils.data.DataLoader(
		train_data,
		batch_size=8, shuffle=True,
		num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(
		test_data,
		batch_size=8, shuffle=True,
		num_workers=4, pin_memory=True)
	return train_loader, test_loader


def load_dataset(data_dir, dataset_name, model):
	if dataset_name == 'LabelMe':
		return load_labelme(data_dir, dataset_name, model)
	elif dataset_name == 'CIFAR-10':
		return load_cifar(data_dir, dataset_name, model)