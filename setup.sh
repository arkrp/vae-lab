#!bin/bash
echo "launching vae-lab."
if [ $BASH_SOURCE = $0 ]; then
    echo "program must be run as 'source setup.sh'"
else
    if ! [ -e "venv" ]; then
        echo "venv not found. Summoning venv!"
        python3 -m venv venv
        venv/bin/pip install -r requirements.txt
    fi
    export PYTHONPATH=$(pwd)/src
    source venv/bin/activate
    echo "vae-lab operaional!"
fi
