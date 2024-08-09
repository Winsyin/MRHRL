import torch
import numpy as np
from model import *
from sklearn import metrics
import pickle
import random
import csv

csv.field_size_limit(500 * 1024 * 1024)

def ReadMyCsv4(SavaDict, fileName):
	csv_reader = csv.reader(open(fileName))
	count = 0
	for row in csv_reader:
		SavaDict[row[0]] = count
		count = count + 1
	return


def ReadMyCsv5(SaveList, fileName):
	csv_reader = csv.reader(open(fileName))
	for row in csv_reader:
		c = int(row[0]) - 1
		m = idMiRNA[row[1]]
		cmi = [c, m]
		SaveList.append(cmi)
	return


def MyLabel(Sample):
	label = []
	for i in range(int(len(Sample) / 2)):
		label.append(1)
	for i in range(int(len(Sample) / 2)):
		label.append(0)
	return label




def load_hyg(filename):
	with open(filename, 'rb') as f:
		hyg_dict = pickle.load(f)

	hy_cm = hyg_dict['hy_cm']
	hy_mc = hyg_dict['hy_mc']
	hy_cc = hyg_dict['hy_cc_top5']
	hy_mm = hyg_dict['hy_mm_top5']

	lg_c = hyg_dict['lg_c2']
	lg_m = hyg_dict['lg_m2']


	return hy_cm, hy_mc, hy_cc, hy_mm, \
		   torch.from_numpy(lg_c).type(torch.FloatTensor), torch.from_numpy(lg_m).type(torch.FloatTensor)


def train(circ_fea, mi_fea, train_data, label_train, hy_ls, c_mat, m_mat):
	model.train()
	print('--- Start training ---')
	optimizer.zero_grad()
	_, pred_train, z_circ_train, z_mi_train = model(circ_fea, mi_fea, train_data[:, 0], train_data[:, 1], hy_ls)

	rec_circ_train = z_circ_train @ z_circ_train.T
	rec_mi_train = z_mi_train @ z_mi_train.T

	loss_1_train = loss_function1(pred_train.view(-1, 1), torch.from_numpy(label_train).view(-1, 1).float())

	loss_rec_c_train = loss_function2(rec_circ_train, c_mat)
	loss_rec_m_train = loss_function2(rec_mi_train, m_mat)

	loss_2_train = 1 * loss_rec_c_train + 1 * loss_rec_m_train
	loss_train = loss_1_train + 0.4 * loss_2_train
	loss_train.backward()
	optimizer.step()


	score_train_cpu = pred_train.detach().numpy()
	auc_train = metrics.roc_auc_score(label_train, score_train_cpu)

	pred_label_train = [1 if j > 0.5 else 0 for j in score_train_cpu]
	acc_train = metrics.accuracy_score(label_train, pred_label_train)

	print('Epoch: {:05d},'.format(e + 1), 'loss_train:{:.6f}'.format(loss_train.item()),
		  'AUC_train:{:.6f},'.format(auc_train), 'ACC_train:{:.6f},'.format(acc_train))


def test(circ_fea, mi_fea, test_data, label_test, hy_ls):
	with torch.no_grad():
		model.eval()
		print('--- Start valuating ---')

		_, pred_test, _, _ = model(circ_fea, mi_fea, test_data[:, 0], test_data[:, 1], hy_ls)


		score_test_cpu = pred_test.detach().numpy()
		auc_test = metrics.roc_auc_score(label_test, score_test_cpu)

		pred_label_test = [1 if j > 0.5 else 0 for j in score_test_cpu]
		acc_test = metrics.accuracy_score(label_test, pred_label_test)


		print('AUC_test:{:.6f},'.format(auc_test), 'ACC_test:{:.6f},'.format(acc_test))

		return score_test_cpu


if __name__=='__main__':
	random.seed(1)
	np.random.seed(1)
	torch.manual_seed(1)

	epochs = 70
	lr = 0.001
	bio_out_dim = 32
	hgnn_dim = 512


	for fold in range(5):
		print('***********fold_{}*****************'.format(fold))


		CMI_data = './5fold_CV/'

		PositiveSample_Train = []
		ReadMyCsv5(PositiveSample_Train, '{}/Positive_Sample_Train{}.csv'.format(CMI_data, fold))
		PositiveSample_Validation = []
		ReadMyCsv5(PositiveSample_Validation, '{}/Positive_Sample_Validation{}.csv'.format(CMI_data, fold))
		PositiveSample_Test = []
		ReadMyCsv5(PositiveSample_Test, '{}/Positive_Sample_Test{}.csv'.format(CMI_data, fold))

		NegativeSample_Train = []
		ReadMyCsv5(NegativeSample_Train, '{}/Negative_Sample_Train{}.csv'.format(CMI_data, fold))
		NegativeSample_Validation = []
		ReadMyCsv5(NegativeSample_Validation, '{}/Negative_Sample_Validation{}.csv'.format(CMI_data, fold))
		NegativeSample_Test = []
		ReadMyCsv5(NegativeSample_Test, '{}/Negative_Sample_Test{}.csv'.format(CMI_data, fold))


		x_train_pair = []
		x_train_pair.extend(PositiveSample_Train)
		x_train_pair.extend(NegativeSample_Train)
		x_train_pair = np.array(x_train_pair)

		x_validation_pair = []
		x_validation_pair.extend(PositiveSample_Validation)
		x_validation_pair.extend(NegativeSample_Validation)
		x_validation_pair = np.array(x_validation_pair)

		x_test_pair = []
		x_test_pair.extend(PositiveSample_Test)
		x_test_pair.extend(NegativeSample_Test)
		x_test_pair = np.array(x_test_pair)

		y_train = MyLabel(x_train_pair)
		y_validation = MyLabel(x_validation_pair)
		y_test = MyLabel(x_test_pair)

		CircEmbeddingFeature = np.loadtxt('./similarity/circRNA_seq_similarity_circRNA2vec.txt', delimiter='\t')
		miRNAEmbeddingFeature = np.loadtxt('./similarity/miRNA_seq_similarity_kmer.txt', delimiter='\t')
		CircEmbeddingFeature = torch.from_numpy(CircEmbeddingFeature).type(torch.FloatTensor)
		miRNAEmbeddingFeature = torch.from_numpy(miRNAEmbeddingFeature).type(torch.FloatTensor)


		hy_cm, hy_mc, hy_cc, hy_mm, circRNA_mat, miRNA_mat = load_hyg('./generate_hyg/hyg_dict{}.pickle'.format(fold))

		model = MRHRL(BioEncoder(CircEmbeddingFeature.shape[0], miRNAEmbeddingFeature.shape[0], bio_out_dim),
					 HgnnEncoder(bio_out_dim, hgnn_dim),
					 Decoder((hgnn_dim // 4) * 2))


		loss_function1 = torch.nn.BCELoss()
		loss_function2 = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr)

		for e in range(epochs):
			train(CircEmbeddingFeature, miRNAEmbeddingFeature,
				x_train_pair, np.array(y_train), [hy_cm, hy_mc, hy_cc, hy_mm], circRNA_mat, miRNA_mat)
			ModelTestOutput = test(CircEmbeddingFeature, miRNAEmbeddingFeature,
							x_validation_pair, np.array(y_validation), [hy_cm, hy_mc, hy_cc, hy_mm])


		ModelTestOutput = test(CircEmbeddingFeature, miRNAEmbeddingFeature,
							x_test_pair, np.array(y_test), [hy_cm, hy_mc, hy_cc, hy_mm])

