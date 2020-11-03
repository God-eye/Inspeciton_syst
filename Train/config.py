############### Variable initialization ##############
class Config():
	'''
	Docstring : This class will initialize all the parameters we need for model training.
	'''
	def __init__(self, train_path, test_path, model_path, train_anom, img_size = (128, 128), batch_size = 3, mx_frm = 4000, stride = [1,2], frm_cnt = 20, test_size = 400, epochs = 10):
	    self.train_path = train_path
	    self.test_path = test_path
	    self.img_size = img_size
	    self.batch_size = batch_size
	    self.model_path = model_path
	    self.train_anom = train_anom
	    self.epochs = epochs
	    self.stride = stride
	    self.mx_frm = mx_frm
	    self.frm_cnt = frm_cnt
	    self.test_size = test_size