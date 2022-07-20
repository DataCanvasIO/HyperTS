from hyperts.tests import skip_if_not_tf

@skip_if_not_tf
class Test_TF_GPU():

    def test_tf_gpu_memory_growth(self):
        from hyperts.utils import tf_gpu
        tf_gpu.set_memory_growth()

    def test_tf_gpu_memory_limit(self):
        from hyperts.utils import tf_gpu
        tf_gpu.set_memory_limit(100)