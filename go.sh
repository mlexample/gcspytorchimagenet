set -eu
export NUM_EPOCHS=${NUM_EPOCHS:-2}
export TPU_NAME=${TPU_NAME:-mytpu}
export BATCH_SIZE=${BATCH_SIZE:-256}
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
if ! gsutil ls ${IMAGE_DIR}/imagenetindex.json ; then
 python mydataloader.py cache
fi
time python -m torch_xla.distributed.xla_dist --tpu=$TPU_NAME --conda-env=torch-xla-1.7 --env XLA_USE_BF16=1 -- python /tmp/thepackage/test_train_mp_imagenet.py \
    --num_epochs=$NUM_EPOCHS \
    --batch_size=$BATCH_SIZE --num_workers=32 --log_steps=20   \
    --logdir=$LOGDIR    --datadir=$IMAGE_DIR
