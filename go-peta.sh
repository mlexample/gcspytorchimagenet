set -eu
export NUM_EPOCHS=${NUM_EPOCHS:-2}
export TPU_NAME=${TPU_NAME:-mytpu}
export BATCH_SIZE=${BATCH_SIZE:-256}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-64}
export NUM_WORKERS=${NUM_WORKERS:-8}
if [ -z "$BUCKET" ]
then
      echo '$BUCKET variable must be defined'
      exit 1
fi
if [ -z "$IMAGE_DIR" ]
then
      echo '$IMAGE_DIR variable must be defined'
      exit 1
fi

source /anaconda3/etc/profile.d/conda.sh
conda activate torch-xla-1.7

export LOGDIR="${LOGDIR:-gs://${BUCKET}/log-$(date '+%Y%m%d%H%M%S')}"
echo "Profiling data in $LOGDIR"

time python -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME --conda-env=torch-xla-1.7 --env XLA_USE_BF16=1 -- python /tmp/thepackage/test_train_mp_petastorm.py \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE --num_workers=$NUM_WORKERS --log_steps=200 \
    --logdir=$LOGDIR --datadir=$IMAGE_DIR --test_set_batch_size=$TEST_BATCH_SIZE \
    --metrics_debug