import unittest
import numpy as np
from GeoInvers import LocalApproach

class TestLocalApproachFunctions(unittest.TestCase):
  def setUp(self):
    self.func = lambda x, y, z: (x + y) / z
    m1 = [1, 2, 3, 4]
    m2 = [1, 2, 3, 4]
    m3 = [1, 2, 3, 4]
    self.params = [m1, m2, m3]
    self.obs_data = [self.func(*param) for param in zip(*self.params)]
    self.deltas = [0.1, 0.2, 0.4]
    self.local_approach = LocalApproach(self.func, self.obs_data, self.params)

  
  def test_initialization_valid(self):
    # test inisialisasi dengan input valid
    la = LocalApproach(self.func, [1, 2, 3], [[1, 2, 3], [1, 2, 3]])
    self.assertIsInstance(la, LocalApproach)
  
  
  def test_initialization_invalid_func(self):
    # test inisialisasi dengan func yang tidak callacle
    with self.assertRaises(ValueError) as context:
      LocalApproach(None, [1, 2, 3], [[1, 2, 3], [1, 2, 3]])

    self.assertEqual(str(context.exception), "func must be a callable function")
  

  def test_initialization_invalid_obs_data(self):
    # test inisialisasi dengan obs_data yang bukan list
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, "invalid_data_type", [[1, 2, 3], [1, 2, 3]])

    self.assertEqual(str(context.exception), "obs_data must be a list")
  

  def test_initialization_invalid_params(self):
    # test inisialisasi dengan params yang bukan list of lists
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, [1, 2, 3], "invalid_params_type")

    self.assertEqual(str(context.exception), "params must be a list")


  def test_initialization_invalid_each_param(self):
    # test inisialisasi dengan setiap param yang bukan list
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, [1, 2, 3], [1, 2, 3])

    self.assertEqual(str(context.exception), "Each parameter in params must be a list")
  

  def test_fit_inputs_invalid_methods(self):
    #  test methods yang tidak ada di ['lm', 'gs', 'gr', 'qn', 'svd']
    # Arrange
    method = "test"
    methods = ['lm', 'gs', 'gr', 'qn', 'svd']

    # Act Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(method=method)

    self.assertEqual(str(context.exception), f"method must be one of {methods}")


  def test_fit_inputs_invalid_damping(self):
    #  test tipe data damping
    # Arrange
    damping = "test"

    # Act Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(damping=damping)

    self.assertEqual(str(context.exception), "damping must be a number")


  def test_fit_inputs_invalid_err_min(self):
    # test tipe data err_min
    # Arrange
    err_min = "test"

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(err_min=err_min)
      
    self.assertEqual(str(context.exception), "err_min must be a number")

  
  def test_fit_inputs_invalid_iter_max(self):
    # test tipe data iter_max
    # Arrange
    iter_max = "test"

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(iter_max=iter_max)
      
    self.assertEqual(str(context.exception), "iter_max must be an integer")


  def test_fit_inputs_invalid_h_list(self):
    # test tipe data h_list
    # Arrange
    h_list = "test"

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(h_list=h_list)
      
    self.assertEqual(str(context.exception), "h_list must be a list or a number")


  def test_fit(self):
    # test method fite
    # Arrage
    params = np.array(self.params)

    # Act
    result = self.local_approach.fit(
      method="lm",
      damping=0.01,
      err_min=0.01,
      iter_max=1,
      h_list=[0.1, 0.2, 0.4]
    )

    # Assert
    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.shape, params.shape)


  def test_jacobian(self):
    # test method __jacobian dengan mengakses menggunakan nama mangling
    # Arrange
    expected_result = np.array([[1., 1., -2.38095238],
                                [0.5, 0.5, -1.04166667],
                                [0.33333333, 0.33333333, -0.67873303],
                                [0.25, 0.25, -0.50505051]])

    # Act
    # First, ensure the fit method is called to initialize attributes properly
    self.local_approach.fit(
      method="lm",
      damping=0.01,
      err_min=0.01,
      iter_max=10,
      h_list=[0.1, 0.2, 0.4]
    )
    jacobian_matrix = self.local_approach._LocalApproach__jacobian()

    # Assert
    np.testing.assert_almost_equal(jacobian_matrix, expected_result, decimal=6)
    

  def test_update_params_invalid_method(self):
      # Test pengecualian untuk metode inversion yang tidak valid
      with self.assertRaises(ValueError):
          self.local_approach._LocalApproach__update_params("invalid_method", np.array([1]), np.array([[1]]))


  def test_rmse(self):
      # Test metode __rmse
      delta_d = np.array([1.0, 1.0, 1.0])
      rmse = self.local_approach._LocalApproach__rmse(delta_d)
      expected_rmse = np.sqrt(np.sum(delta_d**2) / len(delta_d))
      self.assertAlmostEqual(rmse, expected_rmse, places=5)


  # still wrong
  def test_jacobian_length_mismatch(self):
      # Test pengecualian untuk length mismatch di __jacobian
      self.local_approach._LocalApproach__h_list = np.array([0.1])  # Mengubah h_list agar tidak sesuai
      with self.assertRaises(ValueError):
          self.local_approach._LocalApproach__jacobian()
  

if __name__ == '__main__':
  unittest.main() # pragma: no cover