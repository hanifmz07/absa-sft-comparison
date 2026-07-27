@echo off
setlocal enabledelayedexpansion

call .venv\Scripts\activate.bat

rem Specifiy cuda device if needed
set CUDA_VISIBLE_DEVICES=0

set "LANGUAGE=%~1"
if "%LANGUAGE%"=="" (
    echo Error: No language specified.
    exit /b 1
)
rem Validate the language argument
if /i not "%LANGUAGE%"=="indo" if /i not "%LANGUAGE%"=="eng" if /i not "%LANGUAGE%"=="sunda" if /i not "%LANGUAGE%"=="jav" if /i not "%LANGUAGE%"=="mad" if /i not "%LANGUAGE%"=="sun" if /i not "%LANGUAGE%"=="min" (
    echo Error: Invalid language specified. Use 'indo', 'eng', 'sunda', 'jav', 'mad', 'sun', or 'min'.
    exit /b 1
)

set "DATASET_TYPE=%~2"
if "%DATASET_TYPE%"=="" (
    echo Error: Dataset type must be specified.
    exit /b 1
)

set "DATASET_FOLDER=%~3"
if "%DATASET_FOLDER%"=="" (
    echo Error: Dataset folder must be specified. Name a folder located in the hotel_dataset/{lang} directory.
    exit /b 1
)

set "BATCH_SIZE=%~4"
if "%BATCH_SIZE%"=="" (
    echo Error: Batch size must be specified.
    exit /b 1
)

rem Seeds for the SFT process
set SEEDS=9584 123 2024 31415 777

set "LOG_BASE_NAME=sft_full"
set "LOG_DIR=logs"
set "PID=%RANDOM%"
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%T"

echo Running ABSA SFT

for %%S in (%SEEDS%) do (
    set "SEED=%%S"
    echo.
    echo --------------------------------------------------------
    echo Running full sft with seed: !SEED!
    echo --------------------------------------------------------
    echo Processing seed: !SEED!
    echo Processing dataset folder: %DATASET_FOLDER%

    set "PROMPT_TYPE="
    if "%DATASET_FOLDER%"=="mvp_aos" set "PROMPT_TYPE=mvp"
    if "%DATASET_FOLDER%"=="mvp_aos_augment" set "PROMPT_TYPE=mvp"
    if "%DATASET_FOLDER%"=="mvp" set "PROMPT_TYPE=mvp"
    if "%DATASET_FOLDER%"=="gas" set "PROMPT_TYPE=gas"
    if "%DATASET_FOLDER%"=="legoabsa_multitask" set "PROMPT_TYPE=legoabsa"
    if "%DATASET_FOLDER%"=="legoabsa_tasktransfer" set "PROMPT_TYPE=legoabsa"
    if "%DATASET_FOLDER%"=="indolegoabsa_multitask" set "PROMPT_TYPE=legoabsa"

    if "!PROMPT_TYPE!"=="" (
        echo Unknown dataset folder: %DATASET_FOLDER%
    ) else (
        set "LOG_SUBDIR=%LOG_DIR%\%LOG_BASE_NAME%\%DATASET_TYPE%\%LANGUAGE%\%DATASET_FOLDER%\seed_!SEED!"
        if not exist "!LOG_SUBDIR!" mkdir "!LOG_SUBDIR!"
        set "STDOUT_LOG=!LOG_SUBDIR!\%PID%_%TIMESTAMP%.log"
        set "STDERR_LOG=!LOG_SUBDIR!\%PID%_%TIMESTAMP%.err"

        echo. 2>"!STDERR_LOG!"

        powershell -NoProfile -Command "& { python -m src.main.train --train_json_path 'dataset/%DATASET_TYPE%/%LANGUAGE%/%DATASET_FOLDER%/train.json' --model_name 'google/gemma-3-270m' --output_dir 'outputs/models/%DATASET_TYPE%/%LANGUAGE%/%DATASET_FOLDER%/seed_!SEED!/' --prompt_type '!PROMPT_TYPE!' --save_strategy 'epoch' --num_epochs 10 --lr 5e-5 --optimizer 'adamw_torch' --seed !SEED! --batch_size 4 --gradient_accumulation_steps 4 --eval_strategy 'no' } 2>&1 | Tee-Object -FilePath '!STDOUT_LOG!'"
        if errorlevel 1 (
            echo [FAIL] seed !SEED! failed
            exit /b 1
        )
        echo.
    )
)

echo ========================================================
echo All seeds completed at: %DATE% %TIME%
echo ========================================================

endlocal
exit /b 0
