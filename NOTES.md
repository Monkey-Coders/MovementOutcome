### Give permission
chmod +x main.py
### Start script in background
nohup python -u main.py > nohup.log &
### End script
ps ax | grep main.py
kill PID

-- or --

pkill -f main.py