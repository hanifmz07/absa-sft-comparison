@echo off
setlocal enabledelayedexpansion

set "OUTPUT_DIR=%~1"

if "%OUTPUT_DIR%"=="" (
    echo Error: output_dir is required.
    echo Usage: scripts\eval_all_instruct_absa.bat ^<output_dir^>
    exit /b 1
)

set "LANGUAGES=eng indo jav mad min sun"
set "DATASET_TYPE=hotel_reviews"
set "DATASET_FOLDERS=mvp_aos mvp"

for %%L in (%LANGUAGES%) do (
    for %%D in (%DATASET_FOLDERS%) do (
        echo ========================================================
        echo Running instruct-absa eval for: dataset_type=%DATASET_TYPE% lang=%%L dataset_folder=%%D
        echo ========================================================

        call scripts\eval_instruct_absa.bat "%OUTPUT_DIR%" "%DATASET_TYPE%" "%%L" "%%D"
    )
)

endlocal
exit /b 0
