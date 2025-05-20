#!bin/bash
echo "launching vae-lab."
if ! [ -d 'data' ]; then
    echo "data directory not found. Summoning data!"
    mkdir data
fi
if [ $BASH_SOURCE = $0 ]; then
    echo "program must be run as 'source setup.sh'"
else
    if ! [ -e "venv" ]; then
        echo "venv not found. Summoning venv!"
        python3 -m venv venv
        venv/bin/pip install -r requirements.txt
    fi
    export PYTHONPATH=$(pwd)/src
    export DATA_DIR=$(pwd)/data
    source venv/bin/activate
    echo "vae-lab operaional!"
fi
