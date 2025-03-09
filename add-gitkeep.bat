# This is created to add gitkeep in each empy folder

@echo off
for /d /r . %%d in (*) do (
    dir /b "%%d" | findstr "." >nul || echo.>%%d\.gitkeep
)