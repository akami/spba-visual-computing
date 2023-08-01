import os


# TODO documentation
"""
Super-Class for the preprocessing pipeline. Every model that is used
"""
class Model:
    def __init__(self, loader, model_dir, tmp_dir):
        self._loader = loader
        self._model_dir = model_dir
        self._tmp_dir = tmp_dir

    def run(self, *img_paths):
        # step 1: setup
        os.chdir(self._model_dir)
        self._setup()

        # step 2: run model and save image
        img_path = self._run(*img_paths)

        # step 3: teardown (if required)
        self._teardown()

        return img_path

    def _get_img_suffix(self):
        pass

    def _get_pretrained_models(self):
        pass

    def _get_pretrained_model_path(self, model_name):
        pretrained_models = self._get_pretrained_models()

        return os.path.join(
                pretrained_models[model_name]['dir'],
                f"{model_name}.{pretrained_models[model_name]['file_type']}"
        )

    def _setup(self):
        self._loader.download_models(self._get_pretrained_models())

    def _run(self, *img_paths):
        pass

    def _teardown(self):
        pass
