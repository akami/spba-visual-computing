- clone styleGAN3
  - create env using provided environment.yml file with command:
      conda env create -f environment.yml
  - error: ResolvePackageNotFound: cudatoolkit=11.1
- on Windows:
  - use conda-forge channel instead of nvidia (in environment.yml)
  - error persists?
  - check nvcc, gcc (MSVC), cuda toolkit installation and versions

- delete version number of cudatoolkit in environment.yml file
- activate environment: conda activate stylegan3
  - check dependencies: conda list
  - if [cpuonly] remove torch dependencies: conda uninstall * torch *
    - reinstall cuda using command given by https://pytorch.org/get-started/locally/ ; use latest compatible cuda version (>11.1)
    e.g. conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
      - verify gcc compiler is installed and matches recommended version see https://developer.nvidia.com/cuda-toolkit-archive
      - run batch file in base environment such that cl.exe can be found
        - *edit: necessary only once when starting the shell since MSVC 2019 build tools were installed*
        - *note: content of the batch file points to the location of the VC19 batch file and depends on the system architecture*:
        - ```
          & "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64
          ```
          *in general, look for the **vcvarsall.bat** file and copy the path*

- if problem persists: explicitly tell conda to install the cuda version of pytorch: conda install pytorch=*=*cuda* cudatoolkit -c pytorch

- test stylegan3 on windows:
  - python gen_images.py --outdir=out --trunc=1 --seeds=2 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl