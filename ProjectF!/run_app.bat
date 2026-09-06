@echo off
setlocal
title Body posture measurement

rem Try to run the body_measure module using the Python launcher.
py -3 -m body_measure
set "APP_EXIT=%ERRORLEVEL%"

if not "%APP_EXIT%"=="0" (
    echo.
    echo โปรแกรมเริ่มไม่ได้ กรุณาติดตั้ง Python 3.11 หรือใหม่กว่า แล้วรัน:
    echo     py -3 -m pip install -r requirements.txt
    echo.
    echo หากติดตั้งแล้ว ให้ตรวจสอบว่ากล้องไม่ได้ถูกใช้งานโดยโปรแกรมอื่น
    pause
)

exit /b %APP_EXIT%
