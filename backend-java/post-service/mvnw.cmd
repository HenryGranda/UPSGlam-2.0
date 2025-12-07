@REM Maven Wrapper startup script for Windows
@REM 
@echo off

@REM Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set MAVEN_WRAPPER_VERSION=3.2.0
set MAVEN_HOME=%~dp0.mvn\wrapper
set MAVEN_WRAPPER_JAR=%MAVEN_HOME%\maven-wrapper.jar

@REM Execute Maven
%JAVA_HOME%\bin\java -jar "%MAVEN_WRAPPER_JAR%" %*

if ERRORLEVEL 1 goto error
goto end

:error
set ERROR_CODE=1

:end
@REM End local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" endlocal

exit /B %ERROR_CODE%
