import unittest
import numpy as np
from GeoInvers import LocalApproach

class TestLocalApproachFunctions(unittest.TestCase):
  def setUp(self):
    """
    Set up the test environment.
    This method is called before each test case.
    Initializes a sample function, observation data, and parameters for testing.
    """
    self.func = lambda x, y, z: (x + y) / z
    m1 = [1, 2, 3, 4]
    m2 = [1, 2, 3, 4]
    m3 = [1, 2, 3, 4]
    self.params = [m1, m2, m3]
    self.obs_data = [self.func(*param) for param in zip(*self.params)]
    self.deltas = [0.1, 0.2, 0.4]
    self.local_approach = LocalApproach(self.func, self.obs_data, self.params)

  
  def test_initialization_valid(self):
    """
    Test the initialization of LocalApproach with valid inputs.
    Ensures that an instance of LocalApproach is created successfully when given valid parameters.
    """
    la = LocalApproach(self.func, [1, 2, 3], [[1, 2, 3], [1, 2, 3]])
    self.assertIsInstance(la, LocalApproach)
  
  
  def test_initialization_invalid_func(self):
    """
    Test initialization with an invalid function parameter.
    Ensures that a ValueError is raised if the function is not callable.
    """
    with self.assertRaises(ValueError) as context:
      LocalApproach(None, [1, 2, 3], [[1, 2, 3], [1, 2, 3]])

    self.assertEqual(str(context.exception), "func must be a callable function")
  

  def test_initialization_invalid_obs_data(self):
    """
    Test initialization with invalid observation data.
    Ensures that a ValueError is raised if the observation data is not a list.
    """
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, "invalid_data_type", [[1, 2, 3], [1, 2, 3]])

    self.assertEqual(str(context.exception), "obs_data must be a list")
  

  def test_initialization_invalid_params(self):
    """
    Test initialization with invalid parameter data.
    Ensures that a ValueError is raised if the parameters are not a list of lists.
    """
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, [1, 2, 3], "invalid_params_type")

    self.assertEqual(str(context.exception), "params must be a list")


  def test_initialization_invalid_each_param(self):
    """
    Test initialization with parameters where each parameter is not a list.
    Ensures that a ValueError is raised if any element of params is not a list.
    """
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, [1, 2, 3], [1, 2, 3])

    self.assertEqual(str(context.exception), "params must be a list of lists")
  
  
  def test_initialization_invalid_each_param_length(self):
    """
    Test initialization with parameters of different lengths.
    Ensures that a ValueError is raised if the length of each parameter list in params is not the same.
    """
    with self.assertRaises(ValueError) as context:
      LocalApproach(self.func, [1, 2, 3], [[1], [2, 1], [3, 0, 9]])

    self.assertEqual(str(context.exception), "Each parameter in params must be a list of the same length")
  

  def test_fit_inputs_invalid_methods(self):
    """
    Test the fit method with an invalid inversion method.
    Ensures that a ValueError is raised if the method is not one of the allowed methods.
    """
    # Arrange
    method = "test"
    methods = ['lm', 'gs', 'gr', 'qn', 'svd']

    # Act Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(method=method)

    self.assertEqual(str(context.exception), f"method must be one of {methods}")


  def test_fit_inputs_invalid_damping(self):
    """
    Test the fit method with an invalid damping parameter.
    Ensures that a ValueError is raised if damping is not a number.
    """
    # Arrange
    damping = "test"

    # Act Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(damping=damping)

    self.assertEqual(str(context.exception), "damping must be a number")


  def test_fit_inputs_invalid_err_min(self):
    """
    Test the fit method with an invalid err_min parameter.
    Ensures that a ValueError is raised if err_min is not a number.
    """
    # Arrange
    err_min = "test"

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(err_min=err_min)
      
    self.assertEqual(str(context.exception), "err_min must be a number")

  
  def test_fit_inputs_invalid_iter_max(self):
    """
    Test the fit method with an invalid iter_max parameter.
    Ensures that a ValueError is raised if iter_max is not an integer.
    """
    # Arrange
    iter_max = "test"

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(iter_max=iter_max)
      
    self.assertEqual(str(context.exception), "iter_max must be an integer")


  def test_fit_inputs_invalid_h_list_numbers(self):
    """
    Test the fit method with an invalid h_list containing non-numeric values.
    Ensures that a ValueError is raised if any element in h_list is not a number.
    """
    # Arrange
    h_list = ["test"]

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(h_list=h_list)
      
    self.assertEqual(str(context.exception), "All elements in h_list must be numbers")


  def test_fit_inputs_invalid_h_list_number(self):
    """
    Test the fit method with an invalid h_list that is neither a list nor a number.
    Ensures that a ValueError is raised if h_list is not a list of numbers or a single number.
    """
    # Arrange
    h_list = "test"

    # Act & Assert
    with self.assertRaises(ValueError) as context:
      self.local_approach.fit(h_list=h_list)
      
    self.assertEqual(str(context.exception), "h_list must be a list of numbers or a single number")


  def test_fit(self):
    """
    Test the fit method to ensure it returns parameters with the correct shape.
    Ensures that the fit method completes successfully and returns a numpy array of the expected shape.
    """
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
    """
    Test the __jacobian method to ensure it calculates the Jacobian matrix correctly.
    Ensures that the Jacobian matrix computed matches the expected result.
    """
    # Arrange
    expected_result = np.array([[1., 1., -2.38095238],
                                [0.5, 0.5, -1.04166667],
                                [0.33333333, 0.33333333, -0.67873303],
                                [0.25, 0.25, -0.50505051]])

    # Act
    # Ensure the fit method is called to initialize attributes properly
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
    """
    Test the __update_params method with an invalid inversion method.
    Ensures that a ValueError is raised when an unrecognized method is provided.
    """
    with self.assertRaises(ValueError):
        self.local_approach._LocalApproach__update_params("invalid_method", np.array([1]), np.array([[1]]))


  def test_rmse(self):
    """
    Test the __rmse method to ensure it calculates the Root Mean Square Error correctly.
    Ensures that the RMSE value computed matches the expected result.
    """
    delta_d = np.array([1.0, 1.0, 1.0])
    rmse = self.local_approach._LocalApproach__rmse(delta_d)
    expected_rmse = np.sqrt(np.sum(delta_d**2) / len(delta_d))
    self.assertAlmostEqual(rmse, expected_rmse, places=5)


  def test_jacobian_length_mismatch(self):
    """
    Test the __jacobian method with a length mismatch in the h_list parameter.
    Ensures that a ValueError is raised when the length of h_list does not match the number of parameters.
    """
    self.local_approach._LocalApproach__h_list = np.array([0.1])  # Mengubah h_list agar tidak sesuai
    with self.assertRaises(ValueError):
        self.local_approach._LocalApproach__jacobian()
  

if __name__ == '__main__':
  unittest.main() # pragma: no cover