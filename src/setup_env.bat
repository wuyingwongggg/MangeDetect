@echo off

REM Set the environment name from environment.yml or specify manually
FOR /F "tokens=2 delims=: " %%G IN ('findstr /c:"name:" environment.yml') DO SET ENV_NAME=%%G

REM Check if the environment already exists
conda info --envs | find "%ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Environment '%ENV_NAME%' already exists.
) else (
    echo Creating new environment '%ENV_NAME%' from environment.yml...
    conda env create -f environment.yml
)

REM Activate the environment
echo Activating environment '%ENV_NAME%'...
call conda activate %ENV_NAME%

REM Confirm environment activation and list packages
echo Environment '%ENV_NAME%' is now active.
conda list
