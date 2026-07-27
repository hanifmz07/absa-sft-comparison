@echo off
setlocal enabledelayedexpansion

rem Usage:
rem   scripts\eval_all_noconstraint_loop_langs.bat [dataset_type] [dataset_folder] [batch_size] [force]
rem Example (same as your current command, but looped over languages):
rem   scripts\eval_all_noconstraint_loop_langs.bat hotel_reviews mvp 4
rem Pass force=true as the 4th arg to re-evaluate every language even if already evaluated:
rem   scripts\eval_all_noconstraint_loop_langs.bat hotel_reviews mvp 4 true

if "%~1"=="" (set "DATASET_TYPE=hotel_reviews") else (set "DATASET_TYPE=%~1")
if "%~2"=="" (set "DATASET_FOLDER=mvp") else (set "DATASET_FOLDER=%~2")
if "%~3"=="" (set "BATCH_SIZE=4") else (set "BATCH_SIZE=%~3")
if /i not "%~4"=="true" (set "FORCE=false") else (set "FORCE=true")

set "LANGUAGES=eng jav mad min sun"
set "FAILED="

echo Starting looped unconstrained evaluation run
echo dataset_type=%DATASET_TYPE%, dataset_folder=%DATASET_FOLDER%, batch_size=%BATCH_SIZE%, force=%FORCE%
echo languages=%LANGUAGES%
echo ========================================================

for %%L in (%LANGUAGES%) do (
    echo.
    echo [RUN] scripts\eval_all_noconstraint.bat %%L %DATASET_TYPE% %DATASET_FOLDER% %BATCH_SIZE% %FORCE%

    call scripts\eval_all_noconstraint.bat %%L %DATASET_TYPE% %DATASET_FOLDER% %BATCH_SIZE% %FORCE%
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
