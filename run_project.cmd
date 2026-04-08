@echo off
setlocal

rem Wrapper cho phep chay truc tiep bang lenh/file run_project
call "%~dp0run_project.bat" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
