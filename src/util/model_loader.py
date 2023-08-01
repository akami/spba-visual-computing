import bz2
import requests
import os
import gdown

class ModelLoader:

    def download_models(self, model_dic):

        for model in model_dic:

            # get info from model dictionary
            name = model  # key is the name
            dest_dir = model_dic[name]['dir']  # destination folder
            url = model_dic[name]['url']  # url
            file_ending = model_dic[name]['file_type']  # file ending

            # setup destination folder and path
            os.makedirs(dest_dir, exist_ok=True)
            dest_dir = os.path.join(dest_dir, f"{name}.{file_ending}")

            # download file only if it not exists yet
            if not os.path.exists(dest_dir):

                if model_dic[name]['mode'] == 'gdown':
                    gdown.download(url, dest_dir, quiet=False)
                else:
                    self.load_file(url, name, dest_dir)

                # check success
                if os.path.exists(dest_dir):
                    print(f"\nDownload {name} complete!")
                else:
                    print(f"Download {name} failed!")

            # unzip bz2 files
            if file_ending == "dat.bz2":
                self._decompress_bz2(dest_dir)

    def load_file(self, url, name, dest_dir):
        # download file
        response = requests.get(url, stream=True)

        # display progress
        total_size = int(response.headers.get("content-length", 0))
        with open(dest_dir, "wb") as f:
            print(f"Downloading {name} ... \n")
            dl = 0
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    dl += len(chunk)
                    f.write(chunk)
                    self._cli_progress_bar_download(dl, total_size)

    def _decompress_bz2(self, file_path):
        zipfile = bz2.BZ2File(file_path)  # open the file
        data = zipfile.read()  # get the decompressed data
        new_file_path = file_path[:-4]  # assuming the filepath ends with .bz2
        open(new_file_path, 'wb').write(data)

    def _cli_progress_bar_download(self, dl, total_size, width=50):
        """displays download progress"""
        progress = dl / total_size if total_size > 0 else 0
        filled_width = int(width * progress)
        progress_bar_str = '[' + '█' * filled_width + '░' * (width - filled_width) + ']'
        print(f"Progress: {progress_bar_str} {progress * 100:.2f}%", end="\r")
