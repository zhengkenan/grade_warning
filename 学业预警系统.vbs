Set WshShell = CreateObject("WScript.Shell")
' 0 代表隐藏黑框，False 代表不等待它结束
WshShell.Run "cmd /c D:\anaconda\python.exe -m streamlit run D:\grade_warning\app.py", 0, False
' 等待 3 秒，等后台服务起来
WScript.Sleep 3000
' 自动在默认浏览器打开网页
WshShell.Run "http://localhost:8501"