grid run --config gpu_grid_config.yml \
    main.py \
    --architecture 'tcn' \
    --comments 'cond-mix-reverb-overdrive-delay-pitch-noclean' \
    --loss_functions 'esr,mae,stft' \
    --esr_scaling 1 \
    --mae_scaling 1 \
    --stft_scaling 1 \
    --specific_fx_name 'None' \
    --dilation_depth 6 \
    --dilation_factor 7 \
    --kernel_size 20 \
    --activation 'gated' \
    --grouping 'local' \
    --sample_duration 20 \
    --num_channels 16 \
    --gpus -1 \
    --data_dir '/dataset/cond-mixed-reverb-overdrive-delay-pitch-noclean' \
    --batch_size 1 \
    --learning_rate 0.004 \
    --preemphasis_type aw \
    --conditioning \
    --conditioning_type 'deep_film' \
    --conditioning_structure 'shallow' \
    --fx_list 'overdrive,reverb,delay,pitch' \
    # --bias
    # --force_local_residual \
    # --without_preemphasis \
    # --bias
