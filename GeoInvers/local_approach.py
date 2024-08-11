"""
Module GeoInvers
================

This module contains the LocalApproach class, which provides methods for performing 
local optimization approaches to inverse problems using different optimization methods.

Classes:
--------
LocalApproach

LocalApproach
-------------
A class for performing local optimization approaches to solve inverse problems.

Attributes:
-----------
func : Callable
    A function that performs forward modeling and returns the calculated data based on model parameters.
obs_data : np.ndarray
    Observed data for the inverse problem.
params : np.ndarray
    Initial model parameters for the optimization process.
damping : float
    Damping factor used in the Levenberg-Marquardt method.
h_list : np.ndarray
    List of step sizes for central finite difference approximation.

Methods:
--------
__init__(self, func: Callable, obs_data: List[float], params: List[List[float]]) -> None
    Initialize the LocalApproach class with the provided function, observation data, and model parameters.

__check_inputs(self, func, obs_data, params) -> None
    Validate the inputs for the initialization of the class.

__validate_fit_inputs(self, method: str, damping: float, err_min: float, iter_max: int, h_list: Union[List[float], float]) -> None
    Validate the inputs for the fit method.

fit(self, method: str = "lm", damping: float = 0.01, err_min: float = 0.01, iter_max: int = 100, h_list: Union[List[float], float] = 0.01) -> List[float]
    Perform the inversion process using the specified optimization method and parameters.

__inversion(self, method: str, err_min: float, iter_max: int) -> List[float]
    Execute the inversion process to optimize model parameters based on the chosen method.

__update_params(self, method: str, delta_d: np.ndarray, J: np.ndarray) -> np.ndarray
    Update the model parameters based on the specified optimization method.

__lm(self, delta_d: np.ndarray, J: np.ndarray) -> np.ndarray
    Perform the Levenberg-Marquardt optimization to update model parameters.

__rmse(self, delta_d: np.ndarray) -> float
    Calculate the Root Mean Square Error (RMSE) between observed and calculated data.

__jacobian(self) -> np.ndarray
    Compute the Jacobian matrix using central finite differences.
"""

import numpy as np
from numpy.linalg import inv
from copy import deepcopy
from typing import Callable, Union, List

class LocalApproach:

	def __init__(self, func: Callable, obs_data: List[float], params: List[List[float]]) -> None:
		"""
		Initialize the LocalApproach class with the provided function, observation data, and model parameters.

		Parameters
		----------
		func : Callable
				A function that performs forward modeling and returns the calculated data based on model parameters.
		obs_data : List[float]
				Observed data for the inverse problem.
		params : List[List[float]]
				Initial model parameters for the optimization process.

		Returns
		-------
		None
		"""
		self.__check_inputs(func, obs_data, params)

		self.__func = func
		self.__d_obs = np.array(obs_data, dtype=float)
		self.__params = np.array(params, dtype=float)


	def __check_inputs(self, func, obs_data, params) -> None:
		"""
		Validate the inputs for the initialization of the class.

		Parameters
		----------
		func : Callable
				Function to be checked.
		obs_data : List[float]
				Observed data to be checked.
		params : List[List[float]]
				Model parameters to be checked.

		Returns
		-------
		None
		"""
		if not callable(func):
			raise ValueError("func must be a callable function")

		if not isinstance(obs_data, list):
			raise ValueError("obs_data must be a list")
		
		if not isinstance(params, list):
			raise ValueError("params must be a list")

		if len(params) == 0 or not isinstance(params[0], list):
				raise ValueError("params must be a list of lists")

		param_length = len(params[0])
		for param in params:
				if not isinstance(param, list) or len(param) != param_length:
						raise ValueError("Each parameter in params must be a list of the same length")
			
			
	def __validate_fit_inputs(self, method: str, damping: float, err_min: float, iter_max: int, h_list: Union[List[float], float]) -> None:
		"""
		Validate the inputs for the fit method.

		Parameters
		----------
		method : str
				Optimization method to be used ('lm', 'gs', 'gr', 'qn', 'svd').
		damping : float
				Damping factor for the optimization method.
		err_min : float
				Minimum acceptable error for convergence.
		iter_max : int
				Maximum number of iterations.
		h_list : Union[List[float], float]
				List of step sizes for central finite differences or a single value.

		Returns
		-------
		None
		"""
		methods = ['lm', 'gs', 'gr', 'qn', 'svd']
		if method not in methods:
				raise ValueError(f"method must be one of {methods}")
		
		if not isinstance(damping, (float, int)):
				raise ValueError("damping must be a number")
		
		if not isinstance(err_min, (float, int)):
				raise ValueError("err_min must be a number")
		
		if not isinstance(iter_max, int):
				raise ValueError("iter_max must be an integer")
		
		if isinstance(h_list, list):
				if not all(isinstance(h, (float, int)) for h in h_list):
						raise ValueError("All elements in h_list must be numbers")
		elif not isinstance(h_list, (float, int)):
				raise ValueError("h_list must be a list of numbers or a single number")


	def fit(self, method: str = "lm", damping: float = 0.01, err_min: float = 0.01, iter_max: int = 100, h_list: Union[List[float], float] = 0.01) -> List[float]:
		"""
		Perform the inversion process using the specified optimization method and parameters.

		Parameters
		----------
		method : str, optional
				Optimization method to be used ('lm', 'gs', 'gr', 'qn', 'svd'). Default is 'lm'.
		damping : float, optional
				Damping factor for the optimization method. Default is 0.01.
		err_min : float, optional
				Minimum acceptable error for convergence. Default is 0.01.
		iter_max : int, optional
				Maximum number of iterations. Default is 100.
		h_list : Union[List[float], float], optional
				List of step sizes for central finite differences or a single value. Default is 0.01.

		Returns
		-------
		List[float]
				Optimized model parameters after inversion.
		"""
		
		self.__validate_fit_inputs(method, damping, err_min, iter_max, h_list)
		self.__damping = damping
		self.__h_list = np.array(h_list if isinstance(h_list, list) else [h_list] * len(self.__params))
		return self.__inversion(method, err_min, iter_max)


	def __inversion(self, method: str, err_min: float, iter_max: int) -> List[float]:
		"""
		Execute the inversion process to optimize model parameters based on the chosen method.

		Parameters
		----------
		method : str
				Optimization method to be used ('lm', 'gs', 'gr', 'qn', 'svd').
		err_min : float
				Minimum acceptable error for convergence.
		iter_max : int
				Maximum number of iterations.

		Returns
		-------
		List[float]
				Optimized model parameters.
		"""
		old_params = self.__params
		rmse = np.inf
		iteration = 0
		while iteration < iter_max and rmse > err_min:
			iteration += 1
			d_cal = self.__func(*old_params)
			delta_d = (d_cal - self.__d_obs).reshape(-1, 1)
			rmse = self.__rmse(delta_d)
			J = self.__jacobian()
			new_params = self.__update_params(method, delta_d, J)
			old_params = old_params + new_params
		return old_params
	

	def __update_params(self, method: str, delta_d: np.ndarray, J: np.ndarray) -> np.ndarray:
		"""
		Update the model parameters based on the specified optimization method.

		Parameters
		----------
		method : str
				Optimization method to be used ('lm', 'gs', 'gr', 'qn', 'svd').
		delta_d : np.ndarray
				Difference between calculated and observed data.
		J : np.ndarray
				Jacobian matrix.

		Returns
		-------
		np.ndarray
				Updated model parameters.
		"""
		if method == "lm":
				return self.__lm(delta_d, J)
		raise ValueError(f"Unknown method: {method}")
	

	def __lm(self, delta_d: np.ndarray, J: np.ndarray) -> np.ndarray:
		"""
		Perform the Levenberg-Marquardt optimization to update model parameters.

		Parameters
		----------
		delta_d : np.ndarray
				Difference between calculated and observed data.
		J : np.ndarray
				Jacobian matrix.

		Returns
		-------
		np.ndarray
				Updated model parameters.
		"""
		I = np.identity(len(self.__params))
		return inv(J.T @ J + self.__damping**2 * I) @ J.T @ delta_d
	

	def __rmse(self, delta_d: np.ndarray) -> float:
		"""
		Calculate the Root Mean Square Error (RMSE) between observed and calculated data.

		Parameters
		----------
		delta_d : np.ndarray
				Difference between calculated and observed data.

		Returns
		-------
		float
				RMSE value.
		"""
		return np.sqrt(np.sum(delta_d ** 2) / len(delta_d))


	def __jacobian(self) -> np.ndarray:
		"""
		Compute the Jacobian matrix using central finite differences.

		Returns
		-------
		np.ndarray
				Jacobian matrix of partial derivatives.
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
  
