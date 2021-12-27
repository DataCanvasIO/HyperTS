import numpy as np

from hyperts.utils.transformers import (LogXplus1Transformer,
                                        IdentityTransformer,
                                        StandardTransformer,
                                        MinMaxTransformer,
                                        MaxAbsTransformer,
                                        CategoricalTransformer)


class Test_Transformers():

    def get_2d_and_3d_data(self):
        data_2d = np.arange(0, 9).reshape((3, 3))
        data_3d = np.arange(0, 27).reshape((3, 3, 3))

        return data_2d, data_3d

    def scale_tester(self, sc):
        data_2d, data_3d = self.get_2d_and_3d_data()

        transform_data_2d = sc.fit_transform(data_2d)
        inverse_data_2d = sc.inverse_transform(transform_data_2d)

        assert transform_data_2d.shape == (3, 3)
        assert inverse_data_2d.shape == (3, 3)
        assert inverse_data_2d.any() == data_2d.any()

        transform_data_3d = sc.fit_transform(data_3d)
        inverse_data_3d = sc.inverse_transform(transform_data_3d)

        assert transform_data_3d.shape == (3, 3, 3)
        assert inverse_data_3d.shape == (3, 3, 3)
        assert inverse_data_3d.any() == data_3d.any()

        return True

    def test_logxplus1(self):
        sc = LogXplus1Transformer()
        assert self.scale_tester(sc)

    def test_identity(self):
        sc = IdentityTransformer()
        assert self.scale_tester(sc)

    def test_standard(self):
        sc = StandardTransformer()
        assert self.scale_tester(sc)

    def test_minmax(self):
        sc = MinMaxTransformer()
        assert self.scale_tester(sc)

    def test_maxabs(self):
        sc = MaxAbsTransformer()
        assert self.scale_tester(sc)

    def test_ctegorical(self):
        x = np.array(["a", "b", "a", "b", "b"])

        sc = CategoricalTransformer()
        x1 = sc.fit_transform(x)
        x2 = sc.inverse_transform(x1)
        assert x.tolist() == x2.tolist()