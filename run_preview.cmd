@echo off
setlocal

rem Wrapper cho phep chay truc tiep bang lenh/file run_preview
call "%~dp0run_preview.bat" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
