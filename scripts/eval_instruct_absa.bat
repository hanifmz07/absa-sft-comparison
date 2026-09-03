@echo off
setlocal enabledelayedexpansion

call .venv\Scripts\activate.bat

set "OUTPUT_DIR=%~1"
set "DATASET_TYPE=%~2"
set "LANG=%~3"
set "DATASET_FOLDER=%~4"

if "%OUTPUT_DIR%"=="" (
    echo Error: output_dir is required.
    echo Usage: scripts\eval_instruct_absa.bat ^<output_dir^> ^<dataset_type^> ^<lang^> ^<dataset_folder^>
    exit /b 1
)

if "%DATASET_TYPE%"=="" (
    echo Error: dataset_type is required.
    echo Usage: scripts\eval_instruct_absa.bat ^<output_dir^> ^<dataset_type^> ^<lang^> ^<dataset_folder^>
    exit /b 1
)

if "%LANG%"=="" (
    echo Error: lang is required.
    echo Usage: scripts\eval_instruct_absa.bat ^<output_dir^> ^<dataset_type^> ^<lang^> ^<dataset_folder^>
    exit /b 1
)

if "%DATASET_FOLDER%"=="" (
    echo Error: dataset_folder is required.
    echo Usage: scripts\eval_instruct_absa.bat ^<output_dir^> ^<dataset_type^> ^<lang^> ^<dataset_folder^>
    exit /b 1
)

set "SEARCH_ROOT=%OUTPUT_DIR%/evals/%DATASET_TYPE%/%LANG%/%DATASET_FOLDER%"
set "SEARCH_ROOT_WIN=!SEARCH_ROOT:/=\!"

if not exist "!SEARCH_ROOT_WIN!" (
    echo Error: directory not found: !SEARCH_ROOT!
    exit /b 1
)

if /i "%DATASET_FOLDER%"=="mvp" (
    set "PS_INCLUDE='inference_results.json','voting_results.json'"
) else (
    set "PS_INCLUDE='inference_results.json'"
)

set "COUNT=0"
for /f %%N in ('powershell -NoProfile -Command "(Get-ChildItem -Path '!SEARCH_ROOT_WIN!' -Recurse -File -Include !PS_INCLUDE!).Count"') do set "COUNT=%%N"

if "!COUNT!"=="0" (
    echo No result files found under: !SEARCH_ROOT!
    exit /b 0
)

echo Found !COUNT! result file(s) under: !SEARCH_ROOT!

for /f "delims=" %%F in ('powershell -NoProfile -Command "Get-ChildItem -Path '!SEARCH_ROOT_WIN!' -Recurse -File -Include !PS_INCLUDE! | Sort-Object FullName | Select-Object -ExpandProperty FullName"') do (
    echo Processing: %%F
    python -m src.main.eval_instruct_absa --inference_path "%%F"
)

endlocal
exit /b 0
