import os


class Model:
    """
    This class represents the base class for all models used in the pipeline. A model takes an arbitrary number of
    input images and produces an image as a result.
    """
    def __init__(self, loader, model_dir, tmp_dir):
        """
        Constructor of the model class

        :param loader: model loader to load pre-trained models
        :param model_dir: the location of the model in the project
        :param tmp_dir: the directory to store the result
        """
        self._loader = loader
        self._model_dir = model_dir
        self._tmp_dir = tmp_dir

    def run(self, *img_paths):
        """
        The run method of this class provides a guideline for the steps a model needs to go through to produce a result.

        First, we have to change the context in which we run the model, hence we change into the corresponding directory
        in the project. Then, the model goes through the setup process such that it can be run. If necessary, the model
        class also provides a teardown step.

        :param img_paths: input images
        :return: result image path
        """
        # step 1: setup
        os.chdir(self._model_dir)
        self._setup()

        # step 2: run model and save image
        img_path = self._run(*img_paths)

        # step 3: teardown (if required)
        self._teardown()

        return img_path

    def _get_img_suffix(self):
        """
        Information about the processing step. The suffix is added to the name of the result image in the tmp folder.

        :return: image suffix
        """
        pass

    def _get_pretrained_models(self):
        """
        This method returns information about the pre-trained models that need to be downloaded for the model to work.
        A model has to return a dictionary in the following manner, @see ModelLoader:
        {
            'name': {
                'url': the url to download the model from,
                'dir': the directory to store the model,
                'file_type': the file type of the model, e.g. pth,
                'mode': how to download, e.g 'gdown' to use gdown library for google drive links

            }
        }

        :return: Dictionary containing information about needed pre-trained models
        """
        pass

    def _get_pretrained_model_path(self, model_name):
        """
        Returns the path to the pre-trained model according to the dictionary in the _get_pretrained_models() method

        :param model_name: pre-trained model name
        :return: path to pre-trained model
        """
        pretrained_models = self._get_pretrained_models()

        return os.path.join(
                pretrained_models[model_name]['dir'],
                f"{model_name}.{pretrained_models[model_name]['file_type']}"
        )

    def _setup(self):
        """
        Setup step in the run method of the model.
        Per default, the model loader downloads the models specified in the _get_pretrained_models() method in the model
        class.

        :return: void
        """
        self._loader.download_models(self._get_pretrained_models())

    def _run(self, *img_paths):
        """
        Private run method as the second step (after the setup) of the public run method.
        Here, the running the model should be implemented.

        :param img_paths: input image path(s)
        :return: result image path
        """
        pass

    def _teardown(self):
        """
        If necessary, this method implements the teardown/cleanup after the model was run.

        :return: void
        """
        pass
