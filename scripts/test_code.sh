#!/usr/bin/env bash

module="gpssm"
declare -a extras=()  # "experiments" "ploters" "runners")

get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it
          # relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     echo "$DIR"
}

# Change to script root
cd "$(get_script_dir)"/.. || { exit 1; }
GREEN='\033[0;32m'
NC='\033[0m'

# Run style tests
echo -e "${GREEN}Running style tests.${NC}"
flake8 $module --show-source || { exit 1; }
# Ignore import errors for __init__
flake8 $module --filename=__init__.py  --show-source || { exit 1; }

for extra in "${extras[@]}"
do
    flake8 $extra --exclude '__init__.py,test_*.py' --show-source || { exit 1; }
    flake8 $extra --filename=__init__.py  --show-source || { exit 1; }
done


echo -e "${GREEN}Testing docstring conventions.${NC}"
# Test docstring conventions
pydocstyle $module || { exit 1; }
for extra in "${extras[@]}"
do
    pydocstyle $extra || { exit 1; }
done


# Run unit tests
echo -e "${GREEN}Running unit tests.${NC}"

if [ $# -ge 1 ]
then
    ignore_dir=$1
else
    ignore_dir=""
fi
pytest $module --mypy --ignore=$ignore_dir || { exit 1; }
