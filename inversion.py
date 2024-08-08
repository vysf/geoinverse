import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from typing import Callable, Union

class LocalApproach:

	def __init__(self, func: Callable, obs_data: list, params: list[list]) -> None:
		"""
		initiate your data

		Parameters
		----------
		func  : None
				fungsi forward modeling
		obs_data : list
				data observasi
		params : list
				parameter model

		Returns
		-------
		None
		"""
		if not callable(func):
			raise ValueError("func harus sebuag fungsi")

		if not isinstance(obs_data, list):
			raise ValueError("obs_data buat list dong")
		
		if not isinstance(params, list):
			raise ValueError("params buat list dong")

		for param in params:
			if not isinstance(param, list):
				raise ValueError(" parameter model harus lah list")

		self.__func = func
		self.__d_obs = obs_data
		self.__params = params


	def fit(self, method : str = "lm", damping : float = 0.01, err_min : float = 0.01, iter_max : str = 100, h_list: Union[list, float, int] = 0.01) -> tuple:
		"""
		main function in this class

		Parameters
		----------
		method : str
				metode inversi yang digunakan seperti \n
				1. Levemberg-Marquat ('lm')
				2. Gauss-Newton ('gs')
				3. Gradien ('gr)
				4. Quasi-Newton ('qn')
				5. Singular Value Decomposition (svd)
		damping : float
				faktor peturbasi
		err_min : float
				minimum error inversi
		iter_max : int
				maksimum iterasi inversi
		h_list : list or a float
				delta h untuk central finite different
				menyesuaikan banyaknya parameter model

		Returns
		-------
		tuple
			tuple yang berisi beberapa (n parameter) list parameter hasil akhir inversi
		"""
		methods = ['lm', 'gs', 'gr', 'qn', 'svd']
		if method not in methods:
			raise ValueError(f"method harus salah satu dari {methods}")

		if not isinstance(damping, (float, int)):
			raise ValueError("damping harus angka")

		if not isinstance(err_min, (float, int)):
			raise ValueError("error harus angka")

		if not isinstance(iter_max, int):
			raise ValueError("iterasi harus angka")

		if isinstance(h_list, (float, int)):
			h_list = [h_list for h in range(len(self.__params))]

		if not isinstance(h_list, list):
			raise ValueError("h_list harus list")

		for hs in h_list:
			if not isinstance(hs, (float, int)):
				raise ValueError("h harus angka")

		return self.__inversion(method, err_min, iter_max)


	def __inversion(self, method, err_min, iter_max):
		d_obs = self.__d_obs
		old_params = self.__params
		rmse = np.inf
		iteration = 0
		while True:
			iteration += 1

			d_cal = self.__func(*old_params)
			delta_d = (d_cal - d_obs).reshape(-1, 1)

			rmse = self.__rmse(delta_d)

			if iteration == iter_max or rmse <= err_min:
				break

			J = self.__jacobian()

			if method == "lm":
				new_params = self.__lm(delta_d, J)
			
			old_params = old_params + new_params

		return old_params
	
	def __lm(self, delta_d, J):
		I = np.identity(len(self.__params))
		epsilon = 0.01
		params = inv(J.T @ J + epsilon**2 * I) @ J.T @ delta_d
		return params
	

	def __rmse(self, delta_d):
		rmse = np.sqrt(np.sum(delta_d) ** 2 / len(delta_d))
		return rmse


	def __jacobian(self, func1, *args, h_list):
		"""
		Menghitung matriks Jacobian dengan memilih metode perhitungan.
		Menghitung beda hingga tengah untuk fungsi dengan parameter dinamis.
		Args:
		func1: Fungsi yang akan dihitung beda hingganya.
		*args: Daftar nilai untuk setiap parameter.
		h_list: Daftar langkah kecil (step sizes) untuk setiap parameter.

		Returns:
		tuple of numpy arrays: Hasil perhitungan beda hingga untuk fungsi pada daftar nilai parameter.
		"""
		num_params = len(args)
		if num_params != len(h_list):
			raise ValueError("Jumlah parameter input dan langkah kecil harus sama.")

		# Mengonversi setiap daftar nilai menjadi numpy array
		arrays = [np.array(param) for param in args]

		# Mengecek panjang setiap array
		array_lengths = [len(arr) for arr in arrays]
		if not all(length == array_lengths[0] for length in array_lengths):
			raise ValueError("Semua parameter input harus memiliki panjang yang sama.")

		# Menghitung fungsi pada titik-titik awal
		y_initial = np.array([func1(*params) for params in zip(*arrays)])

		# Inisialisasi matriks Jacobian
		J = np.zeros((len(y_initial), num_params))

		for i in range(len(arrays)):
			# Forward modification
			modified_arrays_forward = [deepcopy(arr) for arr in arrays]
			modified_arrays_forward[i] += h_list[i]
			y_forward = np.array([func1(*params) for params in zip(*modified_arrays_forward)])

			# Backward modification
			modified_arrays_backward = [deepcopy(arr) for arr in arrays]
			modified_arrays_backward[i] -= h_list[i]
			y_backward = np.array([func1(*params) for params in zip(*modified_arrays_backward)])

			# Central difference
			J[:, i] = (y_forward - y_backward) / (2 * h_list[i])
			
		return J
  

# if __name__ == "__main__":
#   def f(*args):
#     for arg in args:
#       print(arg)
#     return 1
#   # lc.fit(f,[[1]],['t'], method='l')
#   params = [[1,2], [3,4]]
#   lc = LocalApproach(f, params)
#   f(*params)