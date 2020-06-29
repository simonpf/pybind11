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
    assert(x.flags["WRITEABLE"])

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
   assert(x.flags["WRITEABLE"])

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
   print(x.flags["WRITEABLE"])
   assert(not x.flags["WRITEABLE"])
