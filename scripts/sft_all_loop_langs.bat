@echo off
setlocal enabledelayedexpansion

rem Usage:
rem   scripts\sft_all_loop_langs.bat [dataset_type] [dataset_folder] [batch_size]
rem Example (same as your current command, but looped over languages):
rem   scripts\sft_all_loop_langs.bat hotel_reviews mvp 4

if "%~1"=="" (set "DATASET_TYPE=hotel_reviews") else (set "DATASET_TYPE=%~1")
if "%~2"=="" (set "DATASET_FOLDER=mvp") else (set "DATASET_FOLDER=%~2")
if "%~3"=="" (set "BATCH_SIZE=4") else (set "BATCH_SIZE=%~3")

set "LANGUAGES=eng jav mad min sun"
set "FAILED="

echo Starting looped SFT run
echo dataset_type=%DATASET_TYPE%, dataset_folder=%DATASET_FOLDER%, batch_size=%BATCH_SIZE%
echo languages=%LANGUAGES%
echo ========================================================

for %%L in (%LANGUAGES%) do (
    echo.
    echo [RUN] scripts\sft_all.bat %%L %DATASET_TYPE% %DATASET_FOLDER% %BATCH_SIZE%

    call scripts\sft_all.bat %%L %DATASET_TYPE% %DATASET_FOLDER% %BATCH_SIZE%
    if errorlevel 1 (
        echo [FAIL] %%L failed
        set "FAILED=!FAILED! %%L"
    ) else (
        echo [OK] %%L completed
    )
)

echo.
echo ========================================================
if "%FAILED%"=="" (
    echo All language runs completed successfully.
    endlocal
    exit /b 0
)

echo Completed with failures for language(s):%FAILED%
endlocal
exit /b 1
