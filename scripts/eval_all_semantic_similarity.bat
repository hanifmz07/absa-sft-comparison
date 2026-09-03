@echo off
setlocal enabledelayedexpansion

call .venv\Scripts\activate.bat

set "OUTPUT_DIR=%~1"
set "EMBEDDING_MODEL_NAME=%~2"

if "%OUTPUT_DIR%"=="" (
    echo Error: output_dir is required.
    echo Usage: scripts\eval_all_semantic_similarity.bat ^<output_dir^> ^<embedding_model_name^>
    exit /b 1
)

if "%EMBEDDING_MODEL_NAME%"=="" (
    echo Error: embedding_model_name is required.
    echo Usage: scripts\eval_all_semantic_similarity.bat ^<output_dir^> ^<embedding_model_name^>
    exit /b 1
)

set "LANGUAGES=eng indo jav mad min sun"
set "DATASET_TYPE=hotel_reviews"
set "DATASET_FOLDERS=mvp_aos mvp"

for %%L in (%LANGUAGES%) do (
    for %%D in (%DATASET_FOLDERS%) do (
        echo ========================================================
        echo Running semantic eval for: dataset_type=%DATASET_TYPE% lang=%%L dataset_folder=%%D
        echo ========================================================

        call scripts\eval_semantic_similarity.bat "%OUTPUT_DIR%" "%DATASET_TYPE%" "%%L" "%%D" "%EMBEDDING_MODEL_NAME%"
    )
)

endlocal
exit /b 0
