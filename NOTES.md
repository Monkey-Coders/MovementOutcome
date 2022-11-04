### Give permission
chmod +x sp2022.py
### Start script in background
nohup python -u sp2022.py > nohup.log &
### End script
ps ax | grep sp2022.py
kill PID

-- or --

pkill -f sp2022.py