@echo off
setlocal enabledelayedexpansion

:: Check if the correct number of arguments are provided
if "%~3"=="" (
    echo Usage: run_3dgs.bat ^<3dgs_root^> ^<input_directory^> ^<viewer_root^>
    pause
    exit /b
)

:: Get parameters from command line arguments
set "gs_root=%~1"
set "input=%~2"
set "viewer_root=%~3"

:: Display the input paths for debugging
echo "%gs_root%"
echo "%input%"
echo "%viewer_root%"

:: Extract the folder name from input path
for %%F in ("%input%") do set "fname=%%~nxF"

:: Create necessary directories if not exist
if not exist "%gs_root%\data" mkdir "%gs_root%\data"
if not exist "%gs_root%\data\%fname%" mkdir "%gs_root%\data\%fname%"
if not exist "%gs_root%\data\%fname%\sparse" mkdir "%gs_root%\data\%fname%\sparse"
if not exist "%gs_root%\data\%fname%\sparse\0" mkdir "%gs_root%\data\%fname%\sparse\0"

:: Copy images into the data folder
xcopy /s /e /i "%input%\*" "%gs_root%\data\%fname%\images\"

:: Change to 3dgs_root directory
cd /d "%gs_root%"
echo "now path %cd%"

:: Run COLMAP feature extraction
colmap feature_extractor --database_path "data\%fname%\database.db" --image_path "data\%fname%\images" --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1

:: Run COLMAP exhaustive matching
colmap exhaustive_matcher --database_path "data\%fname%\database.db"

:: Run COLMAP mapper
colmap mapper --database_path "data\%fname%\database.db" --image_path "data\%fname%\images" --output_path "data\%fname%\sparse"

:: Convert model to BIN format
colmap model_converter --input_path "data\%fname%\sparse\0" --output_path "data\%fname%\sparse\0" --output_type BIN

:: Run the training script
python train.py -s "%gs_root%\data\%fname%"

:: Find the most recently modified folder in the output directory (newest one)
for /f "delims=" %%i in ('dir "%gs_root%\output" /b /ad /o-d') do (
    set "generated_id=%%i"
    goto :found
)

:found
:: Print the result
echo The result is stored in %gs_root%\output\%generated_id%

:: Change to the viewer directory and run the viewer app with the generated ID
cd /d "%viewer_root%\bin"
SIBR_gaussianViewer_app.exe -m "%gs_root%\output\%generated_id%"
