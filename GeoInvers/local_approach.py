import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from typing import Callable, Union, List

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
		self._check_inputs(func, obs_data, params)

		self.__func = func
		self.__d_obs = np.array(obs_data, dtype=float)
		self.__params = np.array(params, dtype=float)

	def _check_inputs(self, func, obs_data, params):
		if not callable(func):
			raise ValueError("func must be a callable function")

		if not isinstance(obs_data, list):
			raise ValueError("obs_data must be a list")
		
		if not isinstance(params, list):
			raise ValueError("params must be a list")

		for param in params:
			if not isinstance(param, list):
				raise ValueError("Each parameter in params must be a list")
			
	def _validate_fit_inputs(self, method: str, damping: float, err_min: float, iter_max: int, h_list: Union[List[float], float]):
		methods = ['lm', 'gs', 'gr', 'qn', 'svd']
		if method not in methods:
				raise ValueError(f"method must be one of {methods}")
		
		if not isinstance(damping, (float, int)):
				raise ValueError("damping must be a number")
		
		if not isinstance(err_min, (float, int)):
				raise ValueError("err_min must be a number")
		
		if not isinstance(iter_max, int):
				raise ValueError("iter_max must be an integer")
		
		if not isinstance(h_list, (list, float, int)):
				raise ValueError("h_list must be a list or a number")

	def fit(self, method: str = "lm", damping: float = 0.01, err_min: float = 0.01, iter_max: int = 100, h_list: Union[List[float], float] = 0.01) -> List[float]:
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
		
		self._validate_fit_inputs(method, damping, err_min, iter_max, h_list)
		
		self.__damping = damping
		self.__h_list = np.array(h_list if isinstance(h_list, list) else [h_list] * len(self.__params))

		return self.__inversion(method, err_min, iter_max)


	def __inversion(self, method: str, err_min: float, iter_max: int) -> List[float]:
		old_params = self.__params
		rmse = np.inf
		iteration = 0
		while iteration < iter_max and rmse > err_min:
			iteration += 1
			d_cal = self.__func(*old_params)
			# d_cal = self.__func(*old_params.T)
			delta_d = (d_cal - self.__d_obs).reshape(-1, 1)
			rmse = self.__rmse(delta_d)
			J = self.__jacobian()

			# if method == "lm":
			# 	new_params = self.__lm(delta_d, J)

			new_params = self.__update_params(method, delta_d, J)
			old_params = old_params + new_params
		return old_params
	
	def __update_params(self, method: str, delta_d: np.ndarray, J: np.ndarray) -> np.ndarray:
		if method == "lm":
				return self.__lm(delta_d, J)
		raise ValueError(f"Unknown method: {method}")
	
	def __lm(self, delta_d: np.ndarray, J: np.ndarray) -> np.ndarray:
		I = np.identity(len(self.__params))
		return inv(J.T @ J + self.__damping**2 * I) @ J.T @ delta_d
	

	def __rmse(self, delta_d: np.ndarray) -> float:
		return np.sqrt(np.sum(delta_d ** 2) / len(delta_d))


	def __jacobian(self):
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
		num_params = len(self.__params)
		if num_params != len(self.__h_list):
			raise ValueError("Jumlah parameter input dan langkah kecil harus sama.")

		# # Mengonversi setiap daftar nilai menjadi numpy array
		# arrays = [np.array(param) for param in self.__params]

		# # Mengecek panjang setiap array
		# array_lengths = [len(arr) for arr in arrays]
		# if not all(length == array_lengths[0] for length in array_lengths):
		# 	raise ValueError("Semua parameter input harus memiliki panjang yang sama.")

		# Inisialisasi matriks Jacobian
		J = np.zeros((len(self.__d_obs), num_params))

		for i in range(num_params):
			# Forward modification
			modified_arrays_forward = [deepcopy(arr) for arr in self.__params]
			modified_arrays_forward[i] += self.__h_list[i]
			y_forward = np.array([self.__func(*params) for params in zip(*modified_arrays_forward)])

			# Backward modification
			modified_arrays_backward = [deepcopy(arr) for arr in self.__params]
			modified_arrays_backward[i] -= self.__h_list[i]
			y_backward = np.array([self.__func(*params) for params in zip(*modified_arrays_backward)])

			# Central difference
			J[:, i] = (y_forward - y_backward) / (2 * self.__h_list[i])
			
		return J
  
