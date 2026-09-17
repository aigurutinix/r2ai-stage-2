@echo off
rem Cho vong Auditor 2 xong roi tu chay tron khau cuoi: gop -> build -> verify -> dem lai.
rem Chay qua Task Scheduler nen song sot ca khi phien Claude Code chet.
setlocal
set "PYTHONIOENCODING=utf-8"
set "VIFINQA_ROOT=E:/ViFinQA"
set "PY=E:\r2ai_guru\.venv\Scripts\python.exe"
set "LOG=E:\r2ai_guru\pipeline\finish5.log"
cd /d E:\r2ai_guru\pipeline

echo [cho] doi DONE_EXIT trong audit_run5.log ... > "%LOG%"
:wait
findstr /C:"DONE_EXIT" audit_run5.log >nul 2>&1 && goto ready
timeout /t 120 /nobreak >nul
goto wait

:ready
echo ===== 1. GOP KET QUA VONG 2 ===== >> "%LOG%"
set "AUDIT_IN=E:\r2ai_guru\pipeline\agent_audit5.json"
"%PY%" merge_audit.py --apply >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== 2. BUILD (USE_AGENT=1) ===== >> "%LOG%"
set "USE_AGENT=1"
set "AGENT_OUT=agent_full_v2.json"
"%PY%" build_submission.py >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== 3. VERIFY ===== >> "%LOG%"
"%PY%" verify_submission.py >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== 4. DEM LAI SO CAU SAI CHAC CHAN (moc: 73) ===== >> "%LOG%"
"%PY%" audit_wrong.py >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo ===== 5. KIEM CHAT ===== >> "%LOG%"
set "AUDIT_IN=E:\r2ai_guru\pipeline\agent_audit5.json"
"%PY%" check_audit.py >> "%LOG%" 2>&1

echo FINISH_DONE >> "%LOG%"
