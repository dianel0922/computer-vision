## Requirements (refference from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) )
This is the hardware and software required to run 3D Gaussian Splatting for Real-Time Radiance Fields. Technically, the viewer has much smaller requirements compared to the the optimizer. I decided to list the 

### Hardware
- windows 11
- An NVIDIA GPU with 24GB VRAM* or more. Preferably an RTX 3080 or better. 

### Software
This is the sofware dependencies you will need installed prior to installing the project. Many of these dependencies are shared with other NeRF projects.
- __Git__ - You will need this to pull the code from GitHub. You can download it [here ](https://git-scm.com/downloads). Follow default installation instructions. You can test to see if you have it already installed by typing ```git --version``` into command prompt
- __Conda__ - I recommend using [Anaconda](https://www.anaconda.com/download) because it's easy to install and manage environments in the future. [MiniConda](https://docs.conda.io/en/latest/miniconda.html) is a great lightweight alternative.
- __CUDA Toolkit__ - this was tested with 11.8. Ensure you are not running 11.6 or 12+. You can download CUDA Toolkit [here](https://developer.nvidia.com/cuda-toolkit-archive) You can check which version of CUDA Toolkit you have installed by typing ```nvcc --version``` into command prompt.
- __Visual Studio 2019 or newer__ - You can download and install it [here](https://visualstudio.microsoft.com/vs/older-downloads/). Make sure you add __Desktop Development with C++__ when installing <br>
- __COLMAP__ - Use the Windows binary, it's easy! You can download it [here](https://github.com/colmap/colmap/releases)
- __ImageMagik__ - This is for preparing your images. Download it [here](https://imagemagick.org/script/download.php)
- __FFMPEG__ - Use this to extract images from video. Download it [here](https://ffmpeg.org/download.html)


## How To Run
cmd order
```shell
env_setting.bat <3dgs_root_parent>
run_3dgs.bat <3dgs_root> <image_directory> <viewers_root>
```
for example

```shell
env_setting.bat C:/
run_3dgs.bat C:/gaussian-splatting/ ./data/fern/ ./viewers/
```

## View Result
```shell
cd <viewers_root>/bin
SIBR_gaussianViewer_app.exe -m <path-to-output>/output/folder-name
```
the output folder will look like
```
output
|---folder-name 
   │  cameras.json
   │  cfg_args
   │  exposure.json
   │  input.ply
   │
   |---point_cloud
       |---iteration_30000
       │      point_cloud.ply
       │
       |---iteration_7000
              point_cloud.ply
```
or you can use any software that can preview point cloud 

> NOTE: \
> You should import all the file above to you software otherwise you will loss info  