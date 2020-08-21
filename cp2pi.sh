source_dir=$1
target_dir=$2

rsync -rav -e ssh \
    --exclude='.git*' \
    --exclude='*.pyc' \
    --exclude='*.swp' \
    --exclude='.pytest_cache*' \
    --exclude='*__pycache__*' \
    --exclude='*.egg-info*' \
    $source_dir $target_dir

