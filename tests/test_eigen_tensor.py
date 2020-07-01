import pytest
import pybind11_tests
import numpy as np
module_names = ["eigen_tensor_3_f_r",
                "eigen_tensor_3_f_c",
                "eigen_tensor_4_f_c",
                "eigen_tensor_4_f_r",
                "eigen_tensor_5_f_r",
                "eigen_tensor_5_f_c",
                "eigen_tensor_3_d_r",
                "eigen_tensor_3_d_c",
                "eigen_tensor_3_i_r",
                "eigen_tensor_3_i_c"]

test_modules = [getattr(pybind11_tests, n) for n in module_names]

def get_ref(array):
    shape = array.shape
    value = np.array(0.0, array.dtype)
    for i in range(len(shape)):
        dims = [1, ] * len(shape)
        dims[i] = shape[i]
        value = value + (np.arange(shape[i], dtype=array.dtype) * 10 ** i).reshape(dims)
    return value

t = test_modules[1]
x = t.get_tensor()

@pytest.mark.parametrize("module", test_modules)
def test_reference_tensor(module):
    """
    This function asserts that the returned Tensor does not memory with the C++-side tensor
    x and that setting a given element changes the value of the returned numpy array.
    """
    x = module.get_tensor()
    x_ref = get_ref(x)
    assert(np.all(np.isclose(x, x_ref)))
    assert(x.flags["C_CONTIGUOUS"] == module.c_contiguous)

    dtype = x.dtype
    index = tuple([0, ] * len(x.shape))
    r = module.get_element(index)
    x[index] = r + 1.0
    r2 = module.get_element(list(index))
    assert(not np.isclose(r2, x[index]))

@pytest.mark.parametrize("module", test_modules)
def test_reference_tensor_ref(module):
   """
   This function checks that the returned TensorRef shares memory with the C++-side tensor
   x and that setting a given element changes the value of the returned numpy array.
   """
   x = module.get_tensor_ref()
   x_ref = get_ref(x)
   assert(np.all(np.isclose(x, x_ref)))
   assert(x.flags["C_CONTIGUOUS"] == module.c_contiguous)

   dtype = x.dtype
   index = [0, ] * len(x.shape)
   r = np.array(np.random.rand(), dtype=dtype)
   x[index] = r
   x0 = module.get_element(index)
   assert(np.isclose(x0, r))

@pytest.mark.parametrize("module", test_modules)
def test_reference_tensor_const_ref(module):
   """
   This function checks that the returned TensorRef shares memory with the C++-side tensor
   x and that setting a given element changes the value of the returned numpy array.
   """
   x = module.get_tensor_const_ref()
   x_ref = get_ref(x)
   assert(np.all(np.isclose(x, x_ref)))
   assert(x.flags["C_CONTIGUOUS"] == module.c_contiguous)

@pytest.mark.parametrize("module", test_modules)
def test_reference_tensor_map(module):
   """
   This function checks that the returned TensorMap shares memory with the C++-side tensor
   x and that setting a given element changes the value of the returned numpy array.
   """
   x = module.get_tensor_map()
   x_ref = get_ref(x)
   assert(np.all(np.isclose(x, x_ref)))
   assert(x.flags["C_CONTIGUOUS"] == module.c_contiguous)

   dtype = x.dtype
   index = [0, ] * len(x.shape)
   r = np.array(np.random.rand(), dtype=dtype)
   x[index] = r
   x0 = module.get_element(index)
   assert(np.isclose(x0, r))

@pytest.mark.parametrize("module", test_modules)
def test_add_element_ref(module):
   """
   This function checks that the returned TensorMap shares memory with the C++-side tensor
   x and that setting a given element changes the value of the returned numpy array.
   """
   dtype = module.get_tensor_map().dtype
   order = "C" if module.c_contiguous else "F"
   sizes = np.random.randint(1, 5, size=module.rank)
   x = np.ones(tuple(sizes), order=order, dtype=dtype)
   indices = (0, ) * module.rank
   r = np.array(1.0, dtype=dtype)
   module.add_element(x, list(indices), r)
   assert(np.isclose(x[indices], r + np.array(1, dtype=dtype)))

@pytest.mark.parametrize("module", test_modules)
def test_add_element_copy(module):
   """
   This test ensures that the right overload is chosen when the provided
   array does not match the expected tensor.
   """
   dtype = module.get_tensor_map().dtype
   order = "F" if module.c_contiguous else "C"
   sizes = np.random.randint(1, 5, size=module.rank)
   x = np.ones(tuple(sizes), order=order, dtype=dtype)
   indices = tuple(np.random.randint(1, 5, size=module.rank) % sizes)
   r = np.array(np.random.randint(666), dtype=dtype)
   module.add_element(x, list(indices), r)
   assert(not np.isclose(x[indices], r + np.array(1, dtype=dtype)))

@pytest.mark.parametrize("module", test_modules)
def test_mul_ref(module):
  """
  This function checks that the returned TensorMap shares memory with the C++-side tensor
  x and that setting a given element changes the value of the returned numpy array.
  """
  dtype = module.get_tensor_map().dtype
  order = "C" if module.c_contiguous else "F"
  sizes = np.random.randint(1, 5, size=module.rank)
  x = np.ones(tuple(sizes), order=order, dtype=dtype)
  indices = np.random.randint(1, 5, size=module.rank) % sizes
  r = np.array(np.random.randint(666), dtype=dtype)
  y = module.mul(x, r)
  assert(np.all(np.isclose(x, y)))

@pytest.mark.parametrize("module", test_modules)
def test_mul_copy(module):
  """
  This function checks that the returned TensorMap shares memory with the C++-side tensor
  x and that setting a given element changes the value of the returned numpy array.
  """
  dtype = module.get_tensor_map().dtype
  order = "F" if module.c_contiguous else "C"
  sizes = np.random.randint(1, 5, size=module.rank)
  x = np.ones(tuple(sizes), order=order, dtype=dtype)
  indices = (0, ) * module.rank
  r = np.array(np.random.randint(666), dtype=dtype)
  y = module.mul(x, r)
  assert(np.all(np.isclose(y, r * x)))
