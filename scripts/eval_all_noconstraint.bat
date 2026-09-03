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
if /i not "%LANGUAGE%"=="indo" if /i not "%LANGUAGE%"=="eng" if /i not "%LANGUAGE%"=="sunda" if /i not "%LANGUAGE%"=="sun" if /i not "%LANGUAGE%"=="jav" if /i not "%LANGUAGE%"=="mad" if /i not "%LANGUAGE%"=="min" (
    echo Error: Invalid language specified. Use 'indo', 'eng', 'sunda', 'sun', 'jav', 'mad', or 'min'.
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

rem Optional 5th arg: pass 'true' to force re-evaluation even if output already exists.
set "FORCE=%~5"
if /i not "%FORCE%"=="true" set "FORCE=false"

rem Seeds for the SFT process
set SEEDS=9584 123 2024 31415 777

set "LOG_BASE_NAME=eval"
set "LOG_DIR=logs"
set "PID=%RANDOM%"
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%T"

echo ========================================================
echo Starting Evaluation script run at: %DATE% %TIME%
echo ========================================================

for %%S in (%SEEDS%) do (
    set "SEED=%%S"
    echo Processing seed: !SEED!

    set "TEST_JSON=dataset/%DATASET_TYPE%/%LANGUAGE%/%DATASET_FOLDER%/test.json"
    set "MODEL_DIR=outputs\models\%DATASET_TYPE%\%LANGUAGE%\%DATASET_FOLDER%\seed_!SEED!"
    set "OUTPUT_DIR=outputs/evals/%DATASET_TYPE%/%LANGUAGE%/%DATASET_FOLDER%/seed_!SEED!"

    set "LOG_SUBDIR=%LOG_DIR%\%LOG_BASE_NAME%\%DATASET_TYPE%\%LANGUAGE%\%DATASET_FOLDER%\seed_!SEED!"
    if not exist "!LOG_SUBDIR!" mkdir "!LOG_SUBDIR!"
    set "STDOUT_LOG=!LOG_SUBDIR!\%PID%_%TIMESTAMP%.log"

    if not exist "!MODEL_DIR!" (
        echo -----------------------------------------------------------
        echo Skipping seed !SEED! — model directory not found: !MODEL_DIR!
        echo -----------------------------------------------------------
    ) else (
        for /d %%M in ("!MODEL_DIR!\*") do (
            set "MODEL_PATH=%%M"
            set "MODEL_NAME=%%~nxM"

            set "LATEST_CHECKPOINT="
            for /f "delims=" %%C in ('powershell -NoProfile -Command "Get-ChildItem -LiteralPath '%%M' -Directory -Filter 'checkpoint-*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"') do set "LATEST_CHECKPOINT=%%C"

            if "!LATEST_CHECKPOINT!"=="" (
                echo -----------------------------------------------------------
                echo Skipping !MODEL_NAME! — no checkpoints found.
                echo -----------------------------------------------------------
            ) else (
                for %%K in ("!LATEST_CHECKPOINT!") do set "CHECKPOINT_NAME=%%~nxK"

                set "PROMPT_TYPE="
                if "%DATASET_FOLDER%"=="mvp_aos" set "PROMPT_TYPE=mvp"
                if "%DATASET_FOLDER%"=="mvp_aos_augment" set "PROMPT_TYPE=mvp"
                if "%DATASET_FOLDER%"=="mvp" set "PROMPT_TYPE=mvp"
                if "%DATASET_FOLDER%"=="gas" set "PROMPT_TYPE=gas"
                if "%DATASET_FOLDER%"=="legoabsa_multitask" set "PROMPT_TYPE=legoabsa"
                if "%DATASET_FOLDER%"=="legoabsa_tasktransfer" set "PROMPT_TYPE=legoabsa"
                if "%DATASET_FOLDER%"=="indolegoabsa_multitask" set "PROMPT_TYPE=legoabsa"

                if "!PROMPT_TYPE!"=="" (
                    echo Error: Unexpected dataset folder '%DATASET_FOLDER%'. Cannot determine prompt type.
                    exit /b 1
                )

                set "EVAL_RESULT_DIR=!OUTPUT_DIR!/!MODEL_NAME!/!CHECKPOINT_NAME!"
                set "EVAL_RESULT_DIR_WIN=!OUTPUT_DIR:/=\!\!MODEL_NAME!\!CHECKPOINT_NAME!"

                set "ALREADY_EVALUATED=false"
                if exist "!EVAL_RESULT_DIR_WIN!\constrained_decoding" set "ALREADY_EVALUATED=true"
                if exist "!EVAL_RESULT_DIR_WIN!\unconstrained_decoding" set "ALREADY_EVALUATED=true"

                if "!ALREADY_EVALUATED!"=="true" if /i not "%FORCE%"=="true" (
                    echo -----------------------------------------------------------
                    echo Skipping !MODEL_NAME! — already evaluated. Pass force=true to re-evaluate.
                    echo -----------------------------------------------------------
                ) else (
                    echo -----------------------------------------------------------
                    echo Evaluating model: !MODEL_NAME! ^(Seed: !SEED!^)
                    echo -----------------------------------------------------------

                    powershell -NoProfile -Command "& { python -m src.main.eval --test_json_path '!TEST_JSON!' --model_path '!LATEST_CHECKPOINT!' --prompt_type '!PROMPT_TYPE!' --output_dir '!EVAL_RESULT_DIR!' --batch_size %BATCH_SIZE% --save_predictions } 2>&1 | Tee-Object -FilePath '!STDOUT_LOG!' -Append"
                    if errorlevel 1 (
                        echo [FAIL] Evaluation failed for !MODEL_NAME! ^(Seed: !SEED!^)
                    )
                    echo.
                )
            )
        )
    )
)

echo.
echo ========================================================
echo All evaluations completed at: %DATE% %TIME%
echo ========================================================

endlocal
exit /b 0
