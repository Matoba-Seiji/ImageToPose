@echo off
setlocal
set "SERVER=%~dp0dist\jade_pose_server\jade_pose_server.exe"
if not exist "%SERVER%" (
    echo Bundled pose server was not found:
    echo %SERVER%
    echo Run tools\build_pose_server.ps1 first.
    pause
    exit /b 1
)
"%SERVER%" --host 127.0.0.1 --port 18765
endlocal
