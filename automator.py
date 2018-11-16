import time
from tf_model import model_generator
import csv

from tqdm import tqdm


class csvgen():
	def __init__(self, num_l=3, start_n=100, inc=25, inc_num=16):
		self.name = (str(num_l) + 'layer_sheet')
		self.num_l = num_l
		self.start_n = start_n
		self.inc = inc
		self.inc_num = inc_num
		self.gen_sheet()

	def gen_sheet(self):
		with open(self.name + '.csv', 'w') as f:
			writer = csv.writer(f)
			for r in range(0, self.inc_num + 1):
				result = calc_avs(n_layers = self.num_l, n_nodes_hl = (self.start_n + (r * self.inc)), av_batchsize=10)
				print('this is the result', result)
				writer.writerow([(self.start_n + (r * self.inc)), result[0], result[1]])
		f.close()

#time wrapper
def  timer(f):
	def timed(*a, **kw):
		ts = time.time()
		result = f(*a, **kw)
		te = time.time()

		return (result, (te - ts))
	return timed

@timer
def generate_model(*a, **kw):
	mg = model_generator(*a, **kw)
	return mg.get_accuracy()

def calc_avs(n_layers, n_nodes_hl, av_batchsize=10):
	x_sum=[0,0]
	for x in tqdm(range(av_batchsize)):
		y_sum = generate_model(n_layers, n_nodes_hl)
		x_sum[0] += y_sum[0]
		x_sum[1] += y_sum[1]
		print(y_sum)
	avacc = x_sum[0] / av_batchsize
	avtime = x_sum[1] / av_batchsize
	#print('av acc: ', avacc, ' avtime: ', avtime)
	return (avacc, avtime)

#print(calc_avs(n_nodes_hl=10, n_layers=20))

def main():
	#print('main!')
	#calc_avs(n_layers=3, n_nodes_hl=5, av_batchsize=1)
	csvgen(num_l = 4, start_n = 100, inc = 10, inc_num = 20)
	

if __name__ == '__main__':
	main()
