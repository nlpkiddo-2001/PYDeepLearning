import numpy as np

BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024

class Tensor:
    """Simple tensor class"""
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad = None


    def __repr__(self):
        "String representation for debugging purpose"
        grad_info = f", requires_grad = {self.requires_grad}" if self.requires_grad else ""
        return f"Tensor(data={self.data}, shape={self.shape}{grad_info})"

    def __str__(self):
        """String numpy data"""
        return f"Tensor {self.data}"

    def numpy(self):
        """Numpy's format data"""
        return self.data

    def memory_footprint(self):
        """Return the total bytes (memory)"""
        return self.data.nbytes

    def __add__(self, other):
        """Addition operation"""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        """Subtraction operation"""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """Multiplication operation"""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """Division operation"""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        """Implementation of matrix multiplication"""
        if not isinstance(other, Tensor):
            raise TypeError(f"Data should be of type Tensor but received data of type {type(other)}")

        if self.shape == () and other.shape == ():
            return Tensor(self.data * other.data)
        elif len(self.shape) == 0 and len(other.shape) == 0:
            return Tensor(self.data * other.data)
        elif len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )

        a = self.data
        b = other.data

        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape

            result_data = np.zeros((M, N), dtype=a.dtype)

            for i in range(M):
                for j in range(N):
                    result_data[i, j] = np.dot(a[i, :], b[:, j])

        else:
            result_data = np.matmul(a, b)

        return Tensor(result_data)


    def __matmul__(self, other):
        """Matrix multiplication of two tensor"""
        return self.matmul(other)

    def __getitem__(self, item):
        """Get the specific item"""
        result_data = self.data[item]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)

        result = Tensor(result_data, requires_grad=self.requires_grad)
        return result

    def reshape(self, *shape):
        """Reshape of the data implementation"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")

            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim

                unknown_dim = self.size // known_size
                new_shape = list(new_shape)
                new_shape[unknown_idx] = unknown_dim
                new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(f"Total elements must match: {self.size} ≠ {target_size}")

        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data, requires_grad=self.requires_grad)

    def transpose(self, dim1, dim2):
        """Transpose the dimension"""
        if dim1 is None and dim2 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-2], axes[-1]
                transposed_data = np.transpose(self.data, axes)


        else:
            if dim1 is None or dim2 is None:
                raise ValueError(f"Both Dimension must be specified")
            axes = list(range(len(self.shape)))
            axes[dim1], axes[dim2] = axes[dim2], axes[dim1]
            transposed_data = np.transpose(self.data, axes)

        return Tensor(transposed_data, requires_grad=self.requires_grad)

    def sum(self, axis=None, keepdims=None):
        """Sum the data"""
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result, requires_grad=self.requires_grad)


    def mean(self, axis=None, keepdims=None):
        """Mean the data"""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result, requires_grad=self.requires_grad)

    def max(self, axis=None, keepdims=None):
        """Max  value of the data"""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result, requires_grad=self.requires_grad)


def test_unit_tensor_creation():
    """🧪 Test Tensor creation with various data types."""
    print("🧪 Unit Test: Tensor Creation...")

    # Test scalar creation
    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.requires_grad == False
    assert scalar.grad is None
    assert scalar.dtype == np.float32

    # Test vector creation
    vector = Tensor([1, 2, 3])
    assert np.array_equal(vector.data, np.array([1, 2, 3], dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.size == 3

    # Test matrix creation
    matrix = Tensor([[1, 2], [3, 4]])
    assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert matrix.shape == (2, 2)
    assert matrix.size == 4

    # Test gradient flag (dormant feature)
    grad_tensor = Tensor([1, 2], requires_grad=True)
    assert grad_tensor.requires_grad == True
    assert grad_tensor.grad is None  # Still None until Module 05

    print("✅ Tensor creation works correctly!")

if __name__ == "__main__":
    test_unit_tensor_creation()




