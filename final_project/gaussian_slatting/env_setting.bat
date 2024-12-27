@echo off
setlocal enabledelayedexpansion

:: Check if the correct number of arguments are provided
if "%~1"=="" (
    echo Usage: env_setting.bat ^<3dgs_root_parent^>
    pause
    exit /b
)

set "gs_root=%~1"
echo "%gs_root%"

cd /d "%gs_root%"
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd /d gaussian-splatting
SET DISTUTILS_USE_SDK=1 
conda env create --file environment.yml
conda activate gaussian_splatting