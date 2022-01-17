from hyperts.utils import tf_gpu

class Test_TF_GPU():

    def test_tf_gpu_memory_growth(self):
        tf_gpu.set_memory_growth()

    def test_tf_gpu_memory_limit(self):
        tf_gpu.set_memory_limit(100)