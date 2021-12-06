from hypernets.conf import configure, Configurable, Bool


@configure()
class Config(Configurable):
    # numeric
    numeric_pipeline_enabled = \
        Bool(True,
             config=True,
             help='detect and encode numeric feature from training data or not.'
             )

    # category
    category_pipeline_enabled = \
        Bool(True,
             config=True,
             help='detect and encode category feature from training data or not.'
             )