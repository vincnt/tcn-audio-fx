python main.py \
    --architecture 'tcn' \
    --comments 'local test reverb cond' \
    --loss_functions 'esr,mae,stft' \
    --esr_scaling 1 \
    --mae_scaling 1 \
    --stft_scaling 1 \
    --specific_fx_name 'reverb' \
    --dilation_depth 6 \
    --dilation_factor 7 \
    --kernel_size 20 \
    --activation 'cond_gated' \
    --grouping 'local' \
    --sample_duration 20 \
    --num_channels 16 \
    --cpu \
    --data_dir '/home/jovyan/cond-reverb' \
    --batch_size 1 \
    --learning_rate 0.004 \
    --preemphasis_type aw \
    --conditioning \
    --conditioning_type 'cond_gated' \
    --conditioning_structure 'deep'
    # --fx_list 'overdrive,reverb' \
    # --bias
    # --force_local_residual \
    # --without_preemphasis \
    # --bias
