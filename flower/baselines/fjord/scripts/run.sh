# RUN_LOG_DIR=${RUN_LOG_DIR:-"exp_logs"}

# pushd ../
# mkdir -p $RUN_LOG_DIR
# seed=123
# echo "Running seed $seed"
# echo "Running without KD ..."
# poetry run python -m fjord.main ++manual_seed=$seed ++cuda=false 2>&1 | tee $RUN_LOG_DIR/wout_kd_$seed.log
# echo "Running with KD ..."
# poetry run python -m fjord.main +train_mode=fjord_kd ++manual_seed=$seed ++cuda=false 2>&1 | tee $RUN_LOG_DIR/w_kd_$seed.log
# echo "Done."
# popd

RUN_LOG_DIR=${RUN_LOG_DIR:-"exp_logs"}

pushd ../
mkdir -p $RUN_LOG_DIR

seed=123
echo "Running seed $seed"

echo "Running without KD ..."
poetry run python -m fjord.main \
  ++manual_seed=$seed ++cuda=false \
  ++num_clients=10 ++sampled_clients=3 ++num_rounds=20 ++batch_size=16 \
  2>&1 | tee $RUN_LOG_DIR/wout_kd_$seed.log

echo "Running with KD ..."
poetry run python -m fjord.main +train_mode=fjord_kd \
  ++manual_seed=$seed ++cuda=false \
  ++num_clients=10 ++sampled_clients=3 ++num_rounds=20 ++batch_size=16 \
  2>&1 | tee $RUN_LOG_DIR/w_kd_$seed.log

echo "Done."
popd
